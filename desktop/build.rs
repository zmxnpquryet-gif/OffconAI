fn main() {
    slint_build::compile("ui/app.slint").expect("failed to compile the Slint UI");

    #[cfg(target_os = "windows")]
    {
        println!("cargo:rerun-if-changed=assets/offcon.ico");
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/offcon.ico");
        res.set("ProductName", "OffconAI Desktop");
        res.set("FileDescription", "OffconAI Desktop — Local-First AI Workspace");
        res.set("CompanyName", "OffconAI");
        if let Err(err) = res.compile() {
            println!("cargo:warning=Could not embed Windows icon/resources: {err}");
        }
    }
}
