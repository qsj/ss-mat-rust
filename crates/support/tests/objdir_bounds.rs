use ss_support::obj_dir::{ObjDir, ObjDirBuilder, ObjRec, BLOCK_RECS};
use std::{fs, path::PathBuf};
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let n=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let p=std::env::temp_dir().join(format!("objdir-bounds-{}-{n}",std::process::id()));
        fs::create_dir(&p).unwrap();Self(p)
    }
    fn clean(&self) {assert!(fs::read_dir(&self.0).unwrap().all(|e|!e.unwrap().file_type().unwrap().is_dir()));}
}
impl Drop for Temp {fn drop(&mut self){fs::remove_dir_all(&self.0).unwrap();}}
fn rec(i:u64)->ObjRec {ObjRec{addr:i,file_offset:i*17,class_idx:(i%97) as u32,len:(i%13) as u32,kind:2,elem:1,flags:7}}
#[test]
fn unique_reference_bytes_and_lookup_many_runs() {
    let t=Temp::new();
    for n in [0,1,4097,100000] {
        let mut b=ObjDirBuilder::new(&t.0).unwrap();let mut records=Vec::new();
        for i in (0..n).rev(){let r=rec(i*3);b.push(r).unwrap();records.push(r);}
        records.sort_unstable_by_key(|r|r.addr);
        let mut expected=Vec::new();for r in &records {expected.extend(r.to_bytes());}
        for r in records.iter().step_by(BLOCK_RECS as usize){expected.extend(r.addr.to_le_bytes());}
        expected.extend((n as u64).to_le_bytes());expected.extend(0x5353_4f42_4a44_4952u64.to_le_bytes());
        let out=t.0.join("out");let d=b.finish_with_sort_budget(&out,131072,4096).unwrap();
        assert_eq!(fs::read(&out).unwrap(),expected);
        for (i,r) in records.iter().enumerate(){assert_eq!(d.get(i as u64).unwrap(),*r);assert_eq!(d.lookup(r.addr).unwrap(),Some((i as u64,*r)));}
        assert!(d.get(n).is_err());assert_eq!(d.lookup(u64::MAX).unwrap(),None);t.clean();
    }
}
#[test]
fn duplicate_boundaries_all_equal_and_cleanup() {
    let t=Temp::new();let out=t.0.join("out");
    for all_equal in [false,true] {
        let mut b=ObjDirBuilder::new(&t.0).unwrap();
        for i in 0..10000 {b.push(rec(if all_equal {4095} else {i})).unwrap();}
        b.push(rec(4095)).unwrap();fs::write(&out,b"unchanged").unwrap();
        let e=b.finish_with_sort_budget(&out,0,4096).err().unwrap();
        assert_eq!(e.kind(),std::io::ErrorKind::InvalidData);assert!(e.to_string().contains("0x0000000000000fff"));
        assert_eq!(fs::read(&out).unwrap(),b"unchanged");t.clean();
    }
}
#[test]
fn corrupt_spill_footer_and_output_errors() {
    let t=Temp::new();let out=t.0.join("out");
    for len in [31,32,33] {
        let b=ObjDirBuilder::new(&t.0).unwrap();
        let dir=fs::read_dir(&t.0).unwrap().find_map(|e|{let e=e.unwrap();e.file_type().unwrap().is_dir().then(||e.path())}).unwrap();
        fs::write(dir.join("spill"),vec![0;len]).unwrap();
        assert_eq!(b.finish_with_sort_budget(&out,0,4096).err().unwrap().kind(),std::io::ErrorKind::InvalidData);t.clean();
    }
    let mut b=ObjDirBuilder::new(&t.0).unwrap();b.push(rec(1)).unwrap();
    assert!(b.finish_with_sort_budget(&t.0.join("missing/out"),0,4096).is_err());t.clean();
    let b=ObjDirBuilder::new(&t.0).unwrap();drop(b);t.clean();
    let mut bytes=Vec::new();bytes.extend(u64::MAX.to_le_bytes());bytes.extend(0x5353_4f42_4a44_4952u64.to_le_bytes());fs::write(&out,&bytes).unwrap();
    assert_eq!(ObjDir::open(&out,0).err().unwrap().kind(),std::io::ErrorKind::InvalidData);
}

#[test]
fn publication_replaces_same_fs_and_preserves_directory_on_failure() {
    let t=Temp::new();let out=t.0.join("out");
    fs::write(&out,b"old contents").unwrap();
    let mut b=ObjDirBuilder::new(&t.0).unwrap();b.push(rec(99)).unwrap();
    let d=b.finish_with_sort_budget(&out,0,4096).unwrap();assert_eq!(d.get(0).unwrap(),rec(99));drop(d);t.clean();
    let directory=t.0.join("destination-directory");fs::create_dir(&directory).unwrap();
    fs::write(directory.join("sentinel"),b"preserve").unwrap();
    let mut b=ObjDirBuilder::new(&t.0).unwrap();b.push(rec(1)).unwrap();
    assert!(b.finish_with_sort_budget(&directory,0,4096).is_err());
    assert_eq!(fs::read(directory.join("sentinel")).unwrap(),b"preserve");
    assert!(fs::read_dir(&t.0).unwrap().all(|e|{let e=e.unwrap();!e.file_name().to_string_lossy().starts_with("objdir-") && !e.file_name().to_string_lossy().starts_with(".objdir-stage-")}));
}
