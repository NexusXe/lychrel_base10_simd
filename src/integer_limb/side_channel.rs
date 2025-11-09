use super::{LB, LHS_PTR, OFFSET, UB};
use crate::{impossible, integer_limb::Limb, iterate::ZIPPER_THREAD};
use std::sync::atomic::Ordering;
use std::thread;

#[inline(always)]
pub fn zip_worker() {
    ZIPPER_THREAD.set(thread::current()).unwrap();
    loop {
        while UB.load(Ordering::Acquire) == 0 {
            thread::park();
        }

        let lhs_ptr = LHS_PTR.load(Ordering::Relaxed); // We already Acquired
        let offset = OFFSET.load(Ordering::Relaxed); // Acquired via DATA_LEN
        let lb = LB.load(Ordering::Relaxed);
        let ub = UB.load(Ordering::Relaxed);
        if lb > ub || ub == 0 {
            impossible!("Incoherent zipper thread lb/ub");
        }

        let rhs_ptr = unsafe { lhs_ptr.add(offset) };

        unsafe {
            Limb::zipper(lhs_ptr, rhs_ptr, lb as usize, ub as usize);
        }

        UB.store(0, Ordering::Release);
    }
}
