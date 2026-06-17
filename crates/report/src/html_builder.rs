const CSS: &str = r#"*{margin:0;padding:0;box-sizing:border-box}
:root{--primary:#2563eb;--primary-hover:#1d4ed8;--danger:#ef4444;--warning:#f59e0b;--success:#10b981;--bg:#f8fafc;--card:#fff;--text-main:#0f172a;--text-muted:#64748b;--border:#e2e8f0;--code-bg:#f1f5f9;--font-sans:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;--font-mono:'JetBrains Mono','Fira Code','SF Mono',Monaco,Consolas,monospace}
html{scroll-behavior:smooth}
body{font-family:var(--font-sans);background-color:var(--bg);color:var(--text-main);line-height:1.6;padding:24px}
.container{max-width:1400px;margin:0 auto}
.report-header{background:linear-gradient(135deg,#1e293b 0%,#0f172a 100%);color:#fff;padding:40px;border-radius:16px;margin-bottom:32px;box-shadow:0 10px 30px -10px rgba(15,23,42,.3)}
.report-header h1{font-size:28px;margin-bottom:12px;font-weight:700;letter-spacing:-.5px}
.report-header .meta{opacity:.85;font-size:14px;font-family:var(--font-mono)}
details{background:var(--card);border-radius:10px;margin-bottom:16px;border:1px solid var(--border);box-shadow:0 1px 3px rgba(0,0,0,.02);overflow:hidden;transition:box-shadow .2s ease}
details:hover{box-shadow:0 4px 12px rgba(0,0,0,.04)}
details details{background:#fafafa;border-radius:8px;margin:10px 16px;box-shadow:none;border:1px solid var(--border)}
details details summary{padding:12px 16px;font-size:13.5px;background-color:#fff}
summary{padding:16px 24px;font-weight:600;font-size:15px;cursor:pointer;list-style:none;display:flex;align-items:center;gap:12px;transition:background .2s,color .2s;user-select:none;background-color:var(--card)}
summary:hover{background-color:#f1f5f9;color:var(--primary)}
summary::before{content:'';display:inline-block;width:6px;height:6px;border-top:2px solid var(--text-muted);border-right:2px solid var(--text-muted);transform:rotate(45deg);transition:transform .2s ease;margin-left:4px;margin-right:4px}
details[open]>summary::before{transform:rotate(135deg)}
details[open]>summary{border-bottom:1px solid var(--border);background-color:#f8fafc}
details[open]>summary+*{margin-top:0}
details br:first-child{display:none}
.SecHdr{font-weight:600;font-size:16px;display:inline-flex;align-items:center;gap:8px;flex-wrap:wrap}
.Bigger{font-size:14.5px}
.TSP_1{text-align:left}
.TAL{text-align:left}
.OvhdColorHigh,.OvhdColorMed,.OvhdColorLow{display:inline-block;padding:4px 12px;border-radius:6px;font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;color:#fff}
.OvhdColorHigh{background-color:var(--danger)}
.OvhdColorMed{background-color:var(--warning);color:#0f172a}
.OvhdColorLow{background-color:var(--success)}
table.dt{width:100%;border-collapse:collapse;font-size:14px;margin:12px 0;background-color:var(--card)}
table.dt th{text-align:right;padding:14px 16px;background-color:#f1f5f9;color:#475569;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.5px;border-bottom:2px solid var(--border)}
table.dt th.TAL{text-align:left}
table.dt td{padding:12px 16px;border-bottom:1px solid var(--border);color:#334155;text-align:right;vertical-align:middle}
table.dt td.TAL{text-align:left}
.C_01{background-color:#f8fafc}
.C_01 td{font-weight:600;color:var(--text-main)}
.C_02{background-color:#fff}
.C_03{background-color:#fafafa}
.C_04{background-color:#f0f9ff}
.C_04 td{background-color:#f0f9ff}
tr:hover td{background-color:#f1f5f9!important}
.scroll-wrap{max-height:550px;overflow-y:auto;border-radius:8px;border:1px solid var(--border);box-shadow:inset 0 2px 4px rgba(0,0,0,.02)}
.scroll-wrap table.dt{margin:0}
.scroll-wrap table.dt th{position:sticky;top:0;z-index:10;box-shadow:0 1px 0 var(--border)}
i-s{color:var(--success);font-size:12px}
t-s{color:#0f766e;font-family:var(--font-mono);font-size:13px;background:#f0fdfa;padding:2px 4px;border-radius:4px}
t-c{color:#b45309;font-weight:600;font-family:var(--font-mono)}
t-f{color:#1d4ed8;font-weight:600;font-family:var(--font-mono)}
t-m{color:#701a75;font-weight:600;font-style:italic;font-family:var(--font-mono)}
cd{font-family:var(--font-mono);font-weight:500;background:var(--code-bg);padding:2px 6px;border-radius:4px;font-size:12.5px;color:#334155;border:1px solid #e2e8f0}
[class^="rc_tab"]{display:block;position:relative;border-left:2px solid #cbd5e1;margin-top:4px;margin-bottom:4px}
rc_tab1{padding-left:10px;margin-left:10px}rc_tab2{padding-left:10px;margin-left:20px}rc_tab3{padding-left:10px;margin-left:30px}rc_tab4{padding-left:10px;margin-left:40px}rc_tab5{padding-left:10px;margin-left:50px}rc_tab6{padding-left:10px;margin-left:60px}rc_tab7{padding-left:10px;margin-left:70px}rc_tab8{padding-left:10px;margin-left:80px}rc_tab9{padding-left:10px;margin-left:90px}rc_tab10{padding-left:10px;margin-left:100px}rc_tab11{padding-left:10px;margin-left:110px}rc_tab12{padding-left:10px;margin-left:120px}rc_tab13{padding-left:10px;margin-left:130px}rc_tab14{padding-left:10px;margin-left:140px}rc_tab15{padding-left:10px;margin-left:150px}rc_tab16{padding-left:10px;margin-left:160px}rc_tab17{padding-left:10px;margin-left:170px}rc_tab18{padding-left:10px;margin-left:180px}rc_tab19{padding-left:10px;margin-left:190px}rc_tab20{padding-left:10px;margin-left:200px}rc_tab21{padding-left:10px;margin-left:210px}rc_tab22{padding-left:10px;margin-left:220px}rc_tab23{padding-left:10px;margin-left:230px}rc_tab24{padding-left:10px;margin-left:240px}
li{position:relative;margin-left:-15px;list-style:none}
li input{position:absolute;left:0;opacity:0;z-index:2;cursor:pointer;height:1em;width:1em;top:0}
li input+ol:before{content:"+";position:absolute;margin-left:-40px;margin-top:-14px;color:var(--text-muted);font-weight:700}
li input:checked+ol:before{content:"-"}
li input+ol>li{display:none;margin-left:-18px!important;padding-left:1px}
li label{cursor:pointer;display:block;padding-left:12px}
li input:checked+ol>li{display:block;margin:0 0 .125em}
li input:checked+ol>li:last-child{margin:0 0 .063em}
button{background-color:var(--primary);color:#fff;border:none;padding:8px 18px;border-radius:6px;cursor:pointer;font-size:13px;font-weight:500;transition:background-color .2s ease,transform .1s ease}
button:hover{background-color:var(--primary-hover)}
button:active{transform:scale(.98)}
.modal{display:none;position:fixed;z-index:200;left:0;top:0;width:100%;height:100%;background-color:rgba(15,23,42,.4);backdrop-filter:blur(4px)}
.modal-content{background-color:var(--card);margin:50px auto;padding:32px;border-radius:16px;width:85%;max-width:1000px;box-shadow:0 25px 50px -12px rgba(0,0,0,.25);max-height:85vh;overflow-y:auto;border:1px solid var(--border)}
.close{color:var(--text-muted);float:right;font-size:28px;font-weight:500;cursor:pointer;line-height:1;transition:color .2s}
.close:hover{color:var(--text-main)}
.report-footer{text-align:center;padding:40px 0;color:var(--text-muted);font-size:13px;border-top:1px solid var(--border);margin-top:40px}
@media(max-width:768px){body{padding:16px}.report-header{padding:24px;border-radius:12px}.report-header h1{font-size:20px}summary{padding:14px 18px;font-size:14px}table.dt th,table.dt td{padding:10px 12px;font-size:13px}}"#;

const JS: &str = r#"function openSection(id){var el=document.getElementById(id);el.open=true;el.scrollIntoView({behavior:'smooth',block:'start'})}
var curModal;
function openModal(mid,ev,nl){var m=document.getElementById(mid);curModal=m;var y=ev.clientY+20;var ys=nl*15+90;if(y+ys>window.innerHeight)y=window.innerHeight-ys;if(y<0)y=0;m.style.paddingTop=y+'px';m.style.display='block'}
function closeModal(mid){document.getElementById(mid).style.display='none'}
window.onclick=function(e){if(e.target==curModal)curModal.style.display='none'};
function sortTable(tid,col){var t=document.getElementById('Table-'+tid),sw=true,dir='desc',sc=0;while(sw){sw=false;var rows=t.rows;for(var i=1;i<rows.length-2;i+=2){var s=false,xs=rows[i].getElementsByTagName('TD')[col].innerText,ys=rows[i+2].getElementsByTagName('TD')[col].innerText;var x=parseInt(xs.replace(/,/g,'')),y=parseInt(ys.replace(/,/g,''));if(xs.indexOf('Kb')!=-1)x*=1024;if(ys.indexOf('Kb')!=-1)y*=1024;if(isNaN(x))x=0;if(isNaN(y))y=0;if(dir=='asc'){if(x>y){s=true;break}}else{if(x<y){s=true;break}}}}if(s){rows[i].parentNode.insertBefore(rows[i+2],rows[i]);rows[i+1].parentNode.insertBefore(rows[i+3],rows[i+1]);sw=true;sc++}else if(sc==0&&dir=='asc'){dir='desc';sw=true}}}}
var ENCODED_STRINGS=[****INSERT_ENCODED_STR_LIST_HERE***];
var ESP="|*/",ESS="|",ESC='c',ESSC='s';
function replEncStrs(id){var el=document.getElementById('sd-'+id);rplRec(el)}
function rplRec(el){var ch=el.children;for(var i=0;i<ch.length;i++){rplEl(ch[i]);rplRec(ch[i])}}
function rplEl(el){var rt='',si=0,t=el.innerHTML,pi=t.indexOf(ESP);while(pi!=-1){rt+=t.substring(si,pi);var ni=pi+3,st=0,nc=t.charAt(ni);if(nc==ESC)st=1;else if(nc==ESSC)st=2;if(st==0){si=ni;rt+=t.substring(pi,ni);pi=t.indexOf(ESP,si);continue}var nei=t.indexOf(ESS,ni+1);if(nei!=-1&&nei-ni<=7){var ns=t.substring(ni+1,nei),n=parseInt(ns);if(!isNaN(n)&&n>=0&&n<=ENCODED_STRINGS.length){var s=ENCODED_STRINGS[n];rt+=(st==1?'<t-c>'+s+'</t-c>':'<t-s>'+s+'</t-s>')}else{rt+=t.substring(pi,nei+1)}si=nei+1}else{si=ni;rt+=t.substring(pi,ni)}pi=t.indexOf(ESP,si)}if(rt.length!=0){rt+=t.substring(si);el.innerHTML=rt}}"#;

/// HTML string builder matching the Java JXRay report format.
/// Provides formatting, tables, reference chain rendering, and modal dialogs.
pub struct HtmlStringBuilder {
    buf: String,
    alternate_row: bool,
    last_col_align_left: bool,
    next_table_id: usize,
}

impl HtmlStringBuilder {
    pub fn new() -> Self {
        Self::with_capacity(512 * 1024)
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            buf: String::with_capacity(cap),
            alternate_row: false,
            last_col_align_left: false,
            next_table_id: 0,
        }
    }

    pub fn alternate_row(&self) -> bool {
        self.alternate_row
    }

    pub fn append_raw(&mut self, s: &str) {
        self.buf.push_str(s);
    }

    pub fn append_line(&mut self, s: &str) {
        self.buf.push_str(s);
        self.buf.push_str("<br>\n");
    }

    pub fn append_break(&mut self) {
        self.buf.push_str("<br>\n");
    }

    pub fn append_formatted(&mut self, s: &str) {
        self.buf.push_str(s);
    }

    pub fn append_details_close_tag(&mut self) {
        self.buf.push_str("</details>\n");
    }

    pub fn prepend_bigger_font(&mut self) {
        self.buf = format!("<font class=\"Bigger\"><br>\n{}", self.buf);
    }

    pub fn append_bigger_font_close(&mut self) {
        self.buf.push_str("</font>");
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn insert_at(&mut self, pos: usize, s: &str) {
        self.buf.insert_str(pos, s);
    }

    /// Begin a table with tooltip headers. Each header is `(label, tooltip)`.
    /// Sortable columns get `onclick="sortTable()"`. Tooltip headers use
    /// `<span title="..."><i-s>ⓘ</i-s></span>`.
    pub fn begin_table_with_tooltips(
        &mut self,
        last_col_left: bool,
        approx_rows: usize,
        headers: &[(&str, &str)],
    ) {
        self.alternate_row = false;
        self.last_col_align_left = last_col_left;
        if approx_rows > 100 {
            self.buf.push_str("<div class=\"scroll-wrap\">\n");
        } else {
            self.buf.push_str("<div>\n");
        }
        let table_id = self.next_table_id;
        self.next_table_id += 1;
        self.buf.push_str(&format!("<table class=\"dt\" id=\"Table-{}\">\n", table_id));

        self.buf.push_str("<tr class=\"C_01\">");
        for (i, (label, tooltip)) in headers.iter().enumerate() {
            let is_last = i == headers.len() - 1;
            if is_last && last_col_left {
                self.buf.push_str("<th class=\"TAL\"><b>&nbsp;");
                self.buf.push_str(label);
                self.buf.push_str("&nbsp;</b></th>");
            } else {
                self.buf.push_str(&format!(
                    "<th onclick=\"sortTable({}, {})\">&nbsp<b>&nbsp<span title=\"{}\">{} &#8593;&#8595; &nbsp;<i-s>&#x24d8;</i-s></span>&nbsp</b></th>",
                    table_id, i, tooltip, label
                ));
            }
        }
        self.buf.push_str("</tr>\n");
    }

    pub fn begin_table(&mut self, last_col_left: bool, approx_rows: usize, headers: &[&str]) {
        self.alternate_row = false;
        self.last_col_align_left = last_col_left;
        if approx_rows > 100 {
            self.buf.push_str("<div class=\"scroll-wrap\">\n");
        } else {
            self.buf.push_str("<div>\n");
        }
        let has_encoded = headers.iter().any(|h| h.contains('|'));
        let table_id = if has_encoded {
            let id = self.next_table_id;
            self.next_table_id += 1;
            Some(id)
        } else {
            None
        };
        if let Some(id) = table_id {
            self.buf.push_str(&format!("<table class=\"dt\" id=\"Table-{}\">\n", id));
        } else {
            self.buf.push_str("<table class=\"dt\">\n");
        }
        if !headers.is_empty() {
            self.append_header_row(headers, table_id);
        }
    }

    fn append_header_row(&mut self, headers: &[&str], table_id: Option<usize>) {
        self.buf.push_str("<tr class=\"C_01\">");
        for (i, h) in headers.iter().enumerate() {
            // Last column may be left-aligned
            if i == headers.len() - 1 && self.last_col_align_left {
                self.buf.push_str("<th class=\"TAL\">");
            } else {
                self.buf.push_str("<th>");
            }
            self.buf.push_str("&nbsp");
            // Check if sortable (contains | separator)
            if let Some(tid) = table_id {
                if let Some(pipe_pos) = h.find('|') {
                    let arrow = "&#8593;&#8595;";
                    self.buf.push_str(&format!(
                        "<th onclick=\"sortTable({}, {})\">",
                        tid, i
                    ));
                    self.buf.push_str(&h[..pipe_pos]);
                    self.buf.push_str(arrow);
                    self.buf.push_str(&h[pipe_pos + 1..]);
                    self.buf.push_str("</th>");
                    continue;
                }
            }
            self.buf.push_str(h);
            if i == headers.len() - 1 && self.last_col_align_left {
                self.buf.push_str("</th>");
            } else {
                self.buf.push_str("</th>");
            }
        }
        self.buf.push_str("</tr>\n");
    }

    pub fn append_table_row(&mut self, cells: &[&str]) {
        let row_class = if self.alternate_row { "C_03" } else { "C_02" };
        self.buf.push_str(&format!("<tr class=\"{}\">", row_class));
        for (i, cell) in cells.iter().enumerate() {
            if i == cells.len() - 1 && self.last_col_align_left {
                self.buf.push_str(&format!("<td class=\"TAL\">&nbsp{}</td>", cell));
            } else {
                self.buf.push_str(&format!("<td>&nbsp{}</td>", cell));
            }
        }
        self.buf.push_str("</tr>\n");
        self.alternate_row = !self.alternate_row;
    }

    pub fn end_table(&mut self) {
        self.buf.push_str("</table>\n");
        self.buf.push_str("</div>\n");
    }

    pub fn render_single_row_table(&mut self, cell: &str, is_header: bool) {
        self.begin_table(true, 1, &[]);
        let row_class = if is_header { "C_01" } else { "C_02" };
        self.buf
            .push_str(&format!("<tr class=\"{}\"><td class=\"TAL\">&nbsp{}</td></tr>\n", row_class, cell));
        self.end_table();
    }

    pub fn append_span_row(&mut self, content: &str, colspan: usize, indent: bool) {
        self.buf.push_str("<tr class=\"C_04\">");
        if colspan == 1 {
            self.buf.push_str("<td class=\"TSP_1\">");
        } else {
            self.buf
                .push_str(&format!("<td colspan=\"{}\" class=\"TAL\">", colspan));
        }
        if indent {
            self.buf.push_str("<rc_tab2>");
        }
        self.buf.push_str(content);
        if indent {
            self.buf.push_str("</rc_tab2>");
        }
        self.buf.push_str("</td></tr>\n");
        self.alternate_row = false;
    }

    pub fn append_reference_path_row(
        &mut self,
        chain: &[String],
        extra: Option<&str>,
        force_collapse: bool,
    ) {
        let row_class = if self.alternate_row { "C_03" } else { "C_02" };
        self.buf.push_str(&format!("<tr class=\"{}\">", row_class));
        self.buf.push_str("<td class=\"TAL\">");

        if chain.len() > 1 || force_collapse {
            self.buf
                .push_str(&format!("<details>\n<summary>&nbsp;&nwarr;{}</summary>\n", chain[0]));
            let mut indent_level = 2;
            let mut i = 1;
            while i < chain.len() {
                if indent_level < 10 {
                    indent_level += 1;
                }
                self.buf
                    .push_str(&format!("<rc_tab{}>", indent_level));
                self.buf.push_str("&nwarr;");
                let mut j = i;
                // Check for repeated patterns (compress reference chain)
                let mut repeat_count = 0;
                while j + 2 < chain.len()
                    && chain[j] == chain[j + 2]
                    && chain.get(j + 1) == chain.get(j + 3)
                {
                    repeat_count += 1;
                    j += 2;
                }
                if repeat_count > 0 {
                    self.buf.push_str("{ ");
                    self.buf.push_str(&chain[j - 2]);
                    self.buf.push_str(&format!("</rc_tab{}>", indent_level));
                    self.buf
                        .push_str(&format!("<rc_tab{}>", indent_level));
                    self.buf.push_str("&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp");
                    self.buf.push_str(&chain[j - 1]);
                    self.buf
                        .push_str(&format!(" }} repeated {} times", repeat_count + 1));
                    j += 1;
                } else {
                    self.buf.push_str(&chain[j]);
                    // Skip long chains
                    if i == 59 && chain.len() > 68 {
                        let skip = chain.len() - 60;
                        self.buf
                            .push_str(&format!("</rc_tab{}>", indent_level));
                        self.buf
                            .push_str(&format!("<rc_tab{}>", indent_level));
                        self.buf
                            .push_str("&nbsp&nbsp&nbsp&nbsp... ");
                        self.buf
                            .push_str(&format!("{} elements skipped...", skip - 1));
                        j = chain.len() - 1;
                    }
                    j += 1;
                }
                self.buf
                    .push_str(&format!("</rc_tab{}>", indent_level));
                i = j;
            }
            if let Some(extra) = extra {
                if !extra.is_empty() {
                    self.buf.push_str(extra);
                    self.buf.push_str("<br>\n");
                }
            }
            self.buf.push_str("</details>\n");
        } else {
            self.buf.push_str("&nbsp&nbsp&nwarr;");
            self.buf.push_str(&chain[0]);
            self.buf.push('\n');
            if let Some(extra) = extra {
                if !extra.is_empty() {
                    self.buf.push_str(extra);
                    self.buf.push_str("<br>\n");
                }
            }
        }

        self.buf.push_str("</td></tr>\n");
        self.alternate_row = !self.alternate_row;
    }

    pub fn append_modal_dialog(&mut self, id: &str, content: &str) {
        self.buf.push_str(&format!(
            "<div id=\"{}\" class=\"modal\">\n  <div class=\"modal-content\">\n    <span class=\"close\" onclick=\"closeModal('{}')\">&times;</span>\n    {}  </div>\n</div>\n",
            id, id, content
        ));
    }

    pub fn wrap_with_html_header(
        &mut self,
        title: &str,
        file_name: &str,
        version: &str,
        dump_date: &str,
        jvm_version: &str,
    ) {
        let file_desc = if !file_name.is_empty() {
            format!("{} file ", file_name)
        } else {
            String::new()
        };

        let page_title = format!("SS-Mat report for {}{}", file_desc, title);

        let mut header = String::with_capacity(8192);
        header.push_str("<html>\n<head>\n<meta charset=\"UTF-8\">\n<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n<title>");
        header.push_str(&page_title);
        header.push_str("</title>\n<style>\n");
        header.push_str(CSS);
        header.push_str("\n</style>\n<script>\n");
        let js_with_strings = JS.replace(
            "****INSERT_ENCODED_STR_LIST_HERE***",
            &crate::encoded_strings::take_array_js(),
        );
        header.push_str(&js_with_strings);
        header.push_str("\n</script>\n</head>\n<body>\n<div class=\"container\">\n");
        header.push_str(&format!(
            "<header class=\"report-header\"><h1>SS-Mat {}Heap Dump Report</h1><div class=\"meta\">{}<br>{} &middot; JVM {}</div></header>\n",
            version, file_desc, dump_date, jvm_version
        ));

        self.buf.insert_str(0, &header);
        self.buf
            .push_str("\n<footer class=\"report-footer\">SS-Mat Heap Dump Analyzer (Rust)</footer>\n</div>\n</body>\n</html>");
    }

    pub fn build(self) -> String {
        self.buf
    }
}

impl Default for HtmlStringBuilder {
    fn default() -> Self {
        Self::new()
    }
}
