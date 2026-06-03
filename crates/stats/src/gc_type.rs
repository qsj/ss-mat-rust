/// GC type detected from JVM info.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GcType {
    Serial,
    Parallel,
    Cms,
    G1,
    Zgc,
    Shenandoah,
    Unknown,
}

impl GcType {
    pub fn from_flag(flag: &str) -> Self {
        match flag {
            "UseSerialGC" => Self::Serial,
            "UseParallelGC" => Self::Parallel,
            "UseConcMarkSweepGC" => Self::Cms,
            "UseG1GC" => Self::G1,
            "UseZGC" => Self::Zgc,
            "UseShenandoahGC" => Self::Shenandoah,
            _ => Self::Unknown,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Serial => "Serial",
            Self::Parallel => "Parallel",
            Self::Cms => "CMS",
            Self::G1 => "G1",
            Self::Zgc => "ZGC",
            Self::Shenandoah => "Shenandoah",
            Self::Unknown => "Unknown",
        }
    }
}
