use std::alloc::{AllocError, Allocator, Layout};
use std::ptr;

#[cfg(target_os = "windows")]
use windows::{
    Win32::{
        Foundation::{CloseHandle, GetLastError, HANDLE, LUID},
        Security::{
            AdjustTokenPrivileges, LUID_AND_ATTRIBUTES, LookupPrivilegeValueW,
            SE_PRIVILEGE_ENABLED, TOKEN_ADJUST_PRIVILEGES, TOKEN_PRIVILEGES, TOKEN_QUERY,
        },
        System::{
            Memory::{
                GetLargePageMinimum, MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_EXTENDED_PARAMETER,
                MEM_EXTENDED_PARAMETER_0, MEM_EXTENDED_PARAMETER_1, MEM_LARGE_PAGES, MEM_RELEASE,
                MEM_RESERVE, MemExtendedParameterAddressRequirements,
                MemExtendedParameterAttributeFlags, PAGE_READWRITE, VirtualAlloc2, VirtualFree,
            },
            Threading::{GetCurrentProcess, OpenProcessToken},
        },
    },
    core::{Result as WinResult, w},
};

#[cfg(target_os = "windows")]
use std::mem::zeroed;

#[derive(Clone, Copy)]
pub struct HugePageAllocator;

#[cfg(target_family = "unix")]
use std::num::NonZeroUsize;

#[cfg(target_family = "unix")]
static mut PAGE_SIZE: NonZeroUsize = NonZeroUsize::new(4 * 1024).unwrap();

impl HugePageAllocator {
    #[cfg(target_os = "windows")]
    #[inline]
    fn enable_memory_lock_privilege(process_handle: HANDLE) -> WinResult<()> {
        use std::mem::zeroed;

        unsafe {
            let mut token_handle: HANDLE = zeroed();

            OpenProcessToken(
                process_handle,
                TOKEN_QUERY | TOKEN_ADJUST_PRIVILEGES,
                &raw mut token_handle,
            )?;

            let mut luid: LUID = zeroed();

            LookupPrivilegeValueW(None, w!("SeLockMemoryPrivilege"), &raw mut luid)?;

            let token_privileges = TOKEN_PRIVILEGES {
                PrivilegeCount: 1,
                Privileges: [LUID_AND_ATTRIBUTES {
                    Luid: luid,
                    Attributes: SE_PRIVILEGE_ENABLED,
                }],
            };

            AdjustTokenPrivileges(
                token_handle,
                false,
                Some(&raw const token_privileges),
                size_of::<TOKEN_PRIVILEGES>() as u32,
                None,
                None,
            )?;

            let last_error = GetLastError();

            CloseHandle(token_handle)?;

            if last_error.is_err() {
                return Err(last_error.to_hresult().into());
            }

            Ok(())
        }
    }

    #[cfg(target_family = "windows")]
    pub fn init() -> Result<Self, Box<dyn std::error::Error>> {
        let process_handle = unsafe { GetCurrentProcess() };
        Self::enable_memory_lock_privilege(process_handle)?;
        Ok(Self)
    }

    #[cfg(target_family = "unix")]
    pub fn init() -> Result<Self, Box<dyn std::error::Error>> {
        unsafe {
            let page_size = libc::sysconf(libc::_SC_PAGESIZE);
            if std::hint::unlikely(page_size == -1) {
                return Err(Box::new(std::io::Error::last_os_error()));
            } else if page_size != 0 {
                PAGE_SIZE = std::num::NonZeroUsize::new(page_size as usize).unwrap_unchecked();
            }
        }
        Ok(Self)
    }

    #[cfg(all(not(target_family = "windows"), not(target_family = "unix")))]
    pub fn init() -> Result<Self> {
        unimplemented!()
    }
}

unsafe impl Allocator for HugePageAllocator {
    #[cfg(target_os = "windows")]
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<ptr::NonNull<[u8]>, AllocError> {
        #[cfg(feature = "1g-pages")]
        const HUGE_PAGE_SIZE_BYTES: usize = 1024 * 1024 * 1024;

        #[cfg(all(debug_assertions, feature = "1g-pages"))]
        {
            let large_page_size = unsafe { GetLargePageMinimum() };
            assert!(HUGE_PAGE_SIZE_BYTES.is_multiple_of(large_page_size));
        }

        let size = layout.size();

        if size == 0 {
            return Ok(ptr::NonNull::slice_from_raw_parts(layout.dangling(), 0));
        }

        unsafe {
            let large_page_size = GetLargePageMinimum();
            let aligned_size = (size.div_ceil(large_page_size) + 1) * large_page_size;
            //let alignment = layout.align().div_ceil(large_page_size) * large_page_size;
            //let alignment = HUGE_PAGE_SIZE_BYTES;

            #[cfg(not(feature = "1g-pages"))]
            let alignment = (layout.align().div_ceil(large_page_size)) * large_page_size;

            #[cfg(feature = "1g-pages")]
            let alignment = (layout.align().div_ceil(HUGE_PAGE_SIZE_BYTES)) * HUGE_PAGE_SIZE_BYTES;

            let mut requirements = MEM_ADDRESS_REQUIREMENTS {
                LowestStartingAddress: zeroed(),
                HighestEndingAddress: zeroed(),
                Alignment: alignment,
            };

            let extended_parameter_1 = MEM_EXTENDED_PARAMETER {
                Anonymous1: MEM_EXTENDED_PARAMETER_0 {
                    _bitfield: MemExtendedParameterAddressRequirements.0 as u64,
                },
                Anonymous2: MEM_EXTENDED_PARAMETER_1 {
                    Pointer: (&raw mut requirements).cast::<std::os::raw::c_void>(),
                },
            };

            let extended_parameter_2 = MEM_EXTENDED_PARAMETER {
                Anonymous1: MEM_EXTENDED_PARAMETER_0 {
                    _bitfield: MemExtendedParameterAttributeFlags.0 as u64,
                },
                #[cfg(feature = "1g-pages")]
                Anonymous2: MEM_EXTENDED_PARAMETER_1 { ULong64: 16u64 },

                #[cfg(not(feature = "1g-pages"))]
                Anonymous2: MEM_EXTENDED_PARAMETER_1 { ULong64: 8u64 },
            };

            #[cfg(feature = "1g-pages")]
            let allocation_size =
                aligned_size.div_ceil(HUGE_PAGE_SIZE_BYTES) * HUGE_PAGE_SIZE_BYTES;

            #[cfg(not(feature = "1g-pages"))]
            let allocation_size = aligned_size;

            let ptr = VirtualAlloc2(
                None,
                None,
                allocation_size,
                MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                PAGE_READWRITE.0,
                Some(&mut [extended_parameter_1, extended_parameter_2]),
            );

            if ptr.is_null() {
                let error = windows::core::Error::from_thread();
                eprintln!("HugePageAlloc failed: {error}");
                return Err(AllocError);
            }

            let slice: &mut [u8] = std::slice::from_raw_parts_mut(ptr.cast(), aligned_size);

            Ok(ptr::NonNull::new(slice).unwrap())
        }
    }

    #[cfg(target_os = "windows")]
    #[inline]
    unsafe fn deallocate(&self, ptr: ptr::NonNull<u8>, _layout: Layout) {
        //eprintln!("Deallocating {:} bytes", _layout.size());
        let result = unsafe { VirtualFree(ptr.as_ptr().cast(), 0, MEM_RELEASE) };
        match result {
            Ok(()) => {}
            Err(error) => {
                panic!("{error}");
            }
        }
    }

    #[cfg(target_family = "unix")]
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<ptr::NonNull<[u8]>, AllocError> {
        use libc::{MADV_HUGEPAGE, aligned_alloc, free, posix_madvise};
        static LARGE_PAGE: NonZeroUsize =
            NonZeroUsize::new(unsafe { PAGE_SIZE.get() } * 1024).unwrap();

        let alignment = layout.align().div_ceil(LARGE_PAGE.get()) * LARGE_PAGE.get();
        let size = layout.size().div_ceil(LARGE_PAGE.get()) * LARGE_PAGE.get();
        let ptr = unsafe { aligned_alloc(alignment, size) };

        if ptr.is_null() {
            return Err(AllocError);
        }

        let result = unsafe { posix_madvise(ptr, size, MADV_HUGEPAGE) };
        if result != 0 && cfg!(debug_assertions) {
            unsafe { free(ptr) };
            return Err(AllocError);
        }

        if let Some(output) = ptr::NonNull::new(ptr::slice_from_raw_parts_mut(ptr as *mut u8, size))
        {
            Ok(output)
        } else {
            Err(AllocError)
        }
    }

    #[cfg(target_family = "unix")]
    #[inline]
    unsafe fn deallocate(&self, ptr: ptr::NonNull<u8>, _layout: Layout) {
        unsafe { libc::free(ptr.as_ptr() as *mut libc::c_void) };
    }
}
