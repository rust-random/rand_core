use crate::word::Word;
use core::hash::Hash;

/// Block buffer
// TODO: manual impl Eq, PartialEq
#[derive(Clone, Debug, Eq)]
pub struct BlockBuffer<W: Word, const N: usize>([W; N]);

impl<W: Word, const N: usize> Default for BlockBuffer<W, N> {
    #[inline]
    fn default() -> Self {
        let mut buf = [W::from_usize(0); N];
        buf[0] = W::from_usize(N);
        Self(buf)
    }
}

impl<W: Word, const N: usize> PartialEq for BlockBuffer<W, N> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.into_array() == other.into_array()
    }
}

impl<W: Word, const N: usize> Hash for BlockBuffer<W, N> {
    #[inline]
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.into_array().hash(state);
    }
}

impl<const N: usize> BlockBuffer<u32, N> {
    /// Implement `next_u64` function using buffer and block generation closure.
    #[inline]
    pub fn next_u64(&mut self, mut generate_block: impl FnMut(&mut [u32; N])) -> u64 {
        use core::mem::replace;

        let buf = &mut self.0;
        let pos = usize::try_from(buf[0]).unwrap();

        let (x, y) = if pos < N - 1 {
            let xy = (buf[pos], buf[pos + 1]);
            buf[0] += 2;
            xy
        } else if pos == N - 1 {
            let x = buf[pos];
            generate_block(buf);
            let y = replace(&mut buf[0], 1);
            (x, y)
        } else {
            generate_block(buf);
            let x = replace(&mut buf[0], 2);
            let y = buf[1];
            (x, y)
        };

        u64::from(y) << 32 | u64::from(x)
    }
}

impl<W: Word, const N: usize> BlockBuffer<W, N> {
    /// Represent block buffer as an array of words.
    ///
    /// This method is inteded only for implementing serialization.
    #[inline]
    pub fn into_array(&self) -> [W; N] {
        let mut buf = self.0;
        let pos = buf[0].into_usize();
        buf[1..pos].fill(W::from_usize(0));
        buf
    }

    /// Try to convert array of words into block buffer.
    ///
    /// This method is inteded only for implementing deserialization.
    #[inline]
    pub fn try_from_array(buf: [W; N]) -> Option<Self> {
        let pos = buf[0].into_usize();
        if pos == 0 || pos > N {
            return None;
        }
        if buf[1..pos].iter().any(|&b| b != W::from_usize(0)) {
            return None;
        }
        Some(Self(buf))
    }

    /// Implement `next_u32/u64` function using buffer and block generation closure.
    #[inline]
    pub fn next_word(&mut self, mut generate_block: impl FnMut(&mut [W; N])) -> W {
        let buf = &mut self.0;
        let pos = buf[0].into_usize();
        debug_assert_ne!(pos, 0, "cursor position should not be zero");
        match buf.get(pos) {
            Some(&val) => {
                buf[0].increment();
                val
            }
            None => {
                generate_block(buf);
                core::mem::replace(&mut buf[0], W::from_usize(1))
            }
        }
    }

    /// Implement `fill_bytes` using buffer and block generation closure.
    #[inline]
    pub fn fill_bytes(&mut self, mut dst: &mut [u8], mut generate_block: impl FnMut(&mut [W; N])) {
        let buf = &mut self.0;
        let word_size = size_of::<W>();

        let pos = buf[0];
        let pos_usize = pos.into_usize();
        debug_assert_ne!(pos_usize, 0, "cursor position should not be zero");
        if pos_usize < buf.len() {
            let buf_tail = &buf[pos_usize..];
            let buf_rem = size_of_val(buf_tail);

            if buf_rem >= dst.len() {
                let new_pos = read_bytes(buf, dst, pos);
                buf[0] = new_pos;
                return;
            }

            let (l, r) = dst.split_at_mut(buf_rem);
            read_bytes(buf, l, pos);
            dst = r;
        }

        let mut blocks = dst.chunks_exact_mut(N * word_size);
        let zero = W::from_usize(0);
        for block in &mut blocks {
            // We intentionally use the temporary buffer to prevent unnecessary writes
            // to the original `buf` and to enable potential optimization of writing
            // generated data directly into `block`.
            let mut buf = [zero; N];
            generate_block(&mut buf);
            read_bytes(&buf, block, zero);
        }

        let rem = blocks.into_remainder();
        let new_pos = if rem.is_empty() {
            W::from_usize(N)
        } else {
            generate_block(buf);
            read_bytes::<W, N>(buf, rem, zero)
        };
        buf[0] = new_pos;
    }
}

/// Reads bytes from `&block[pos..new_pos]` to `dst` using little endian byte order
/// ignoring the tail bytes if necessary and returns `new_pos`.
///
/// This function is written in a way which helps the compiler to compile it down
/// to one `memcpy`. The temporary buffer gets eliminated by the compiler, see:
/// https://rust.godbolt.org/z/T8f77KjGc
#[inline]
fn read_bytes<W: Word, const N: usize>(block: &[W; N], dst: &mut [u8], pos: W) -> W {
    let word_size = size_of::<W>();
    let pos = pos.into_usize();
    assert!(size_of_val(&block[pos..]) >= size_of_val(dst));

    // TODO: replace with `[0u8; { size_of::<W>() * N }]` on
    // stabilization of `generic_const_exprs`
    let mut buf = [W::from_usize(0); N];
    // SAFETY: it's safe to reference `[u32/u64; N]` as `&mut [u8]`
    // with length equal to `size_of::<u32/u64>() * N`
    let buf: &mut [u8] = unsafe {
        let p: *mut u8 = buf.as_mut_ptr().cast();
        let len = word_size * N;
        core::slice::from_raw_parts_mut(p, len)
    };

    for (src, dst) in block.iter().zip(buf.chunks_exact_mut(word_size)) {
        let val = src.to_le_bytes();
        dst.copy_from_slice(val.as_ref())
    }

    let offset = pos * word_size;
    dst.copy_from_slice(&buf[offset..][..dst.len()]);
    let read_words = dst.len().div_ceil(word_size);
    W::from_usize(pos + read_words)
}
