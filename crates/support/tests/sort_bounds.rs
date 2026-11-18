use ss_support::external_sort::ExternalSorter;
use std::{alloc::{GlobalAlloc, Layout, System}, cell::Cell, fs, io::Write, path::PathBuf};

struct Meter;
thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static LIVE: Cell<isize> = const { Cell::new(0) };
    static PEAK: Cell<isize> = const { Cell::new(0) };
}
fn change(n: isize) {
    if ACTIVE.try_with(|v| v.get()).unwrap_or(false) {
        LIVE.with(|v| { let next = v.get() + n; v.set(next); PEAK.with(|p| p.set(p.get().max(next))); });
    }
}
unsafe impl GlobalAlloc for Meter {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l); if !p.is_null() { change(l.size() as isize); } p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) { change(-(l.size() as isize)); System.dealloc(p,l); }
    unsafe fn realloc(&self, p:*mut u8, l:Layout, n:usize)->*mut u8 {
        let q=System.realloc(p,l,n); if !q.is_null() {change(n as isize-l.size() as isize);} q
    }
}
#[global_allocator] static ALLOC: Meter = Meter;
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let n=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let p=std::env::temp_dir().join(format!("sort-bounds-{}-{n}",std::process::id()));
        fs::create_dir(&p).unwrap(); Self(p)
    }
    fn clean(&self) { assert!(fs::read_dir(&self.0).unwrap().all(|e| !e.unwrap().file_name().to_string_lossy().starts_with(".ss-sort-"))); }
}
impl Drop for Temp { fn drop(&mut self) {fs::remove_dir_all(&self.0).unwrap();} }
fn key(b:&[u8])->u64 {u64::from_le_bytes(b[..8].try_into().unwrap())}

#[test]
fn bounded_many_runs_skew_equal_empty_and_reference() {
    let temp=Temp::new(); let input=temp.0.join("input"); let out=temp.0.join("out");
    for (budget,count,equal) in [(68,12000,true),(4096,200000,false),(4096,200000,true),(4096,0,false)] {
        let mut f=fs::File::create(&input).unwrap();
        let mut expected=Vec::new();
        for i in 0..count as u64 {
            let k=if equal {7} else {(i*7919)%200003};
            let record=[k.to_le_bytes(), i.to_le_bytes(), [0;8], [0;8]].concat();
            f.write_all(&record).unwrap(); expected.push(record);
        }
        drop(f);
        LIVE.with(|v|v.set(0)); PEAK.with(|v|v.set(0)); ACTIVE.with(|v|v.set(true));
        let result=ExternalSorter::new(32,budget).sort_file(&input,&out,key);
        ACTIVE.with(|v|v.set(false));
        assert_eq!(result.unwrap(),count);
        let peak=PEAK.with(|v|v.get());
        eprintln!("budget={budget}, input_bytes={}, allocator_peak={peak}",count*32);
        // Paths and formatting have fixed size independent of the run count.
        assert!(peak <= (budget+4096) as isize);
        let bytes=fs::read(&out).unwrap();
        let actual:Vec<_>=bytes.chunks_exact(32).map(|r|r.to_vec()).collect();
        assert!(actual.windows(2).all(|w|key(&w[0])<=key(&w[1])));
        if !equal {expected.sort_unstable_by_key(|r|key(r)); assert_eq!(actual,expected);}
        else {let mut actual=actual; actual.sort_unstable();expected.sort_unstable();assert_eq!(actual,expected);}
        temp.clean();
    }
}
#[test]
fn rejects_ragged_tiny_budget_and_cleans_errors() {
    let temp=Temp::new(); let input=temp.0.join("input");let out=temp.0.join("out");
    fs::write(&input,[0;33]).unwrap();
    assert_eq!(ExternalSorter::new(32,4096).sort_file(&input,&out,key).unwrap_err().kind(),std::io::ErrorKind::InvalidData);
    fs::write(&input,[0;32]).unwrap();
    assert_eq!(ExternalSorter::new(32,67).sort_file(&input,&out,key).unwrap_err().kind(),std::io::ErrorKind::InvalidInput);
    fs::create_dir(&out).unwrap();
    assert!(ExternalSorter::new(32,4096).sort_file(&input,&out,key).is_err());
    temp.clean();
    // Truncate the opened input during run sorting, after its initial metadata check.
    let bytes=vec![0;32000];fs::write(&input,bytes).unwrap();
    let once=Cell::new(false);
    assert!(ExternalSorter::new(32,4096).sort_file(&input,&temp.0.join("changed"), |b| {
        if !once.replace(true) {fs::OpenOptions::new().write(true).open(&input).unwrap().set_len(4096).unwrap();}
        key(b)
    }).is_err());
    temp.clean();
}

#[test]
fn in_place_and_preserved_output_before_publication() {
    let temp = Temp::new();
    let input = temp.0.join("input");
    let output = temp.0.join("output");
    let records: Vec<u8> = (0..1000u64).rev().flat_map(|n| n.to_le_bytes()).collect();
    fs::write(&input, records).unwrap();
    assert_eq!(ExternalSorter::new(8,128).sort_file(&input,&input,key).unwrap(),1000);
    let actual = fs::read(&input).unwrap();
    for (i, rec) in actual.chunks_exact(8).enumerate() { assert_eq!(key(rec), i as u64); }
    fs::write(&output,b"preserve this existing output").unwrap();
    fs::write(&input,vec![0;32000]).unwrap();
    let once=Cell::new(false);
    assert!(ExternalSorter::new(32,4096).sort_file(&input,&output, |b| {
        if !once.replace(true) {fs::OpenOptions::new().write(true).open(&input).unwrap().set_len(4096).unwrap();}
        key(b)
    }).is_err());
    assert_eq!(fs::read(&output).unwrap(),b"preserve this existing output");
    temp.clean();
}
