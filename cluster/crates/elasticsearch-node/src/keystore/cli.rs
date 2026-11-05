//! `elasticsearch-keystore`: ES's keystore tool (`KeyStoreCli` and its
//! subcommands), with the same subcommands, options, prompts, messages and
//! exit codes:
//!
//! ```text
//! elasticsearch-keystore create [-p]
//! elasticsearch-keystore list
//! elasticsearch-keystore add [-f] [-x] <setting>...
//! elasticsearch-keystore add-file [-f] (<setting> <path>)...
//! elasticsearch-keystore remove <setting>...
//! elasticsearch-keystore show <setting>
//! elasticsearch-keystore upgrade
//! elasticsearch-keystore passwd
//! elasticsearch-keystore has-passwd
//! ```
//!
//! The keystore lives in `$ES_PATH_CONF` (default `config`). A protected
//! keystore's password and the values of `add` are read from standard input
//! (without echo on a terminal), one line each, as ES's terminal reads them.

use super::{keystore_path, KeyStore, KeystoreError, PROMPT};
use std::io::Write;
use std::path::{Path, PathBuf};

/// `org.elasticsearch.cli.ExitCodes`.
pub mod exit {
    pub const OK: i32 = 0;
    pub const USAGE: i32 = 64;
    pub const DATA_ERROR: i32 = 65;
    pub const CODE_ERROR: i32 = 70;
    pub const IO_ERROR: i32 = 74;
    pub const CONFIG: i32 = 78;
}

/// `Terminal.Verbosity`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Verbosity {
    Silent,
    Normal,
    Verbose,
}

/// The tool's input and output (a fake one in tests).
pub trait Terminal {
    /// Print `prompt` and read one line (`Terminal.readText`).
    fn read_text(&mut self, prompt: &str) -> Result<String, KeystoreError>;
    /// Print `prompt` and read one line without echo (`Terminal.readSecret`).
    fn read_secret(&mut self, prompt: &str) -> Result<String, KeystoreError>;
    /// Standard output.
    fn out(&mut self, text: &[u8]);
    /// Standard error.
    fn err(&mut self, text: &str);
    /// Whether standard output is redirected (`Terminal.getOutputStream() != null`).
    fn output_is_file(&self) -> bool;
}

/// The process's standard streams.
pub struct SystemTerminal;

impl Terminal for SystemTerminal {
    fn read_text(&mut self, prompt: &str) -> Result<String, KeystoreError> {
        use std::io::BufRead;
        let mut out = std::io::stdout();
        let _ = out.write_all(prompt.as_bytes());
        let _ = out.flush();
        let mut line = String::new();
        let n = std::io::stdin().lock().read_line(&mut line).map_err(|e| KeystoreError::Io(e.to_string()))?;
        if n == 0 {
            return Err(KeystoreError::IllegalState("unable to read from standard input; is standard input open and a tty attached?".into()));
        }
        while line.ends_with('\n') || line.ends_with('\r') {
            line.pop();
        }
        Ok(line)
    }

    fn read_secret(&mut self, prompt: &str) -> Result<String, KeystoreError> {
        use std::io::IsTerminal;
        if std::io::stdin().is_terminal() { super::read_secret(prompt) } else { self.read_text(prompt) }
    }

    fn out(&mut self, text: &[u8]) {
        let mut out = std::io::stdout();
        let _ = out.write_all(text);
        let _ = out.flush();
    }

    fn err(&mut self, text: &str) {
        let _ = std::io::stderr().write_all(text.as_bytes());
    }

    fn output_is_file(&self) -> bool {
        use std::io::IsTerminal;
        !(std::io::stdin().is_terminal() && std::io::stdout().is_terminal())
    }
}

/// A command failure: ES's `UserException` (exit code, optional message) or
/// an exception the command did not handle.
#[derive(Debug)]
enum CliError {
    User(i32, Option<String>),
    Keystore(KeystoreError),
}

impl From<KeystoreError> for CliError {
    fn from(e: KeystoreError) -> CliError {
        match e {
            KeystoreError::Config(m) => CliError::User(exit::CONFIG, Some(m)),
            other => CliError::Keystore(other),
        }
    }
}

fn user(code: i32, msg: impl Into<String>) -> CliError {
    CliError::User(code, Some(msg.into()))
}

const DESCRIPTION: &str = "A tool for managing settings stored in the elasticsearch keystore";

/// The subcommands, in ES's order, with their descriptions.
const COMMANDS: &[(&str, &str)] = &[
    ("create", "Creates a new elasticsearch keystore"),
    ("list", "List entries in the keystore"),
    ("show", "Show a value from the keystore"),
    ("add", "Add string settings to the keystore"),
    ("add-file", "Add a file setting to the keystore"),
    ("remove", "Remove settings from the keystore"),
    ("upgrade", "Upgrade the keystore format"),
    ("passwd", "Changes the password of a keystore"),
    ("has-passwd", "Succeeds if the keystore exists and is password-protected, fails with exit code 1 otherwise."),
];

/// Options each subcommand accepts beyond `-h`, `-s`, `-v` and `-E`:
/// (short, long, description).
fn command_options(cmd: &str) -> &'static [(char, &'static str, &'static str)] {
    match cmd {
        "create" => &[('p', "password", "Prompt for password to encrypt the keystore")],
        "add" => &[
            ('f', "force", "Overwrite existing setting without prompting, creating keystore if necessary"),
            ('x', "stdin", "Read setting values from stdin"),
        ],
        "add-file" => &[('f', "force", "Overwrite existing setting without prompting, creating keystore if necessary")],
        _ => &[],
    }
}

/// Parsed options of one command line.
#[derive(Default, Debug)]
struct Opts {
    flags: Vec<char>,
    args: Vec<String>,
}

impl Opts {
    fn has(&self, c: char) -> bool {
        self.flags.contains(&c)
    }
}

/// jopt-simple style parsing: `-abc` clusters, `--long`, `-E k=v` /
/// `-Ek=v`, `--` ends the options. `posix` stops at the first non-option
/// (the multi-command's own options).
fn parse(args: &[String], extra: &[(char, &str, &str)], posix: bool) -> Result<Opts, String> {
    let mut o = Opts::default();
    let mut i = 0;
    let known = |c: char| matches!(c, 'h' | 's' | 'v') || extra.iter().any(|(s, _, _)| *s == c);
    while i < args.len() {
        let a = &args[i];
        if a == "--" {
            o.args.extend(args[i + 1..].iter().cloned());
            break;
        }
        if let Some(long) = a.strip_prefix("--") {
            let name = long.split('=').next().unwrap_or(long);
            let c = match name {
                "help" => 'h',
                "silent" => 's',
                "verbose" => 'v',
                _ => match extra.iter().find(|(_, l, _)| *l == name) {
                    Some((c, _, _)) => *c,
                    None => return Err(format!("{name} is not a recognized option")),
                },
            };
            o.flags.push(c);
        } else if a.len() > 1 && a.starts_with('-') {
            if let Some(kv) = a.strip_prefix("-E") {
                // `-E key=value`: accepted, settings are not used by this tool.
                if kv.is_empty() {
                    i += 1;
                    if i >= args.len() {
                        return Err("Option E requires an argument".into());
                    }
                }
            } else {
                for c in a[1..].chars() {
                    if !known(c) {
                        return Err(format!("{c} is not a recognized option"));
                    }
                    o.flags.push(c);
                }
            }
        } else {
            if posix {
                o.args.extend(args[i..].iter().cloned());
                break;
            }
            o.args.push(a.clone());
        }
        i += 1;
    }
    if o.has('s') && o.has('v') {
        return Err("Option(s) [v/verbose] are unavailable given other options on the command line".into());
    }
    Ok(o)
}

fn options_help(extra: &[(char, &str, &str)]) -> String {
    let mut rows: Vec<(String, String)> = vec![("-E <KeyValuePair>".into(), "Configure a setting".into())];
    let mut all: Vec<(char, &str, &str)> = extra.to_vec();
    all.extend([('h', "help", "Show help"), ('s', "silent", "Show minimal output"), ('v', "verbose", "Show verbose output")]);
    all.sort_by_key(|(c, _, _)| *c);
    for (c, l, d) in all {
        rows.push((format!("-{c}, --{l}"), d.to_string()));
    }
    let w = rows.iter().map(|(a, _)| a.len()).max().unwrap_or(0).max("Option".len());
    let mut s = format!("{:<w$}  Description\n{:<w$}  -----------\n", "Option", "------");
    for (a, d) in rows {
        s.push_str(&format!("{a:<w$}  {d}\n"));
    }
    s
}

/// One tool invocation: the terminal and its verbosity.
struct Ctx<'a> {
    term: &'a mut dyn Terminal,
    verbosity: Verbosity,
}

impl Ctx<'_> {
    fn println(&mut self, v: Verbosity, text: &str) {
        if v <= self.verbosity {
            self.term.out(format!("{text}\n").as_bytes());
        }
    }

    fn eprintln(&mut self, v: Verbosity, text: &str) {
        if v <= self.verbosity {
            self.term.err(&format!("{text}\n"));
        }
    }

    /// `Terminal.promptYesNo`.
    fn prompt_yes_no(&mut self, prompt: &str, default_yes: bool) -> Result<bool, CliError> {
        loop {
            let suffix = if default_yes { " [Y/n]" } else { " [y/N]" };
            let answer = self.term.read_text(&format!("{prompt}{suffix}"))?;
            if answer.is_empty() {
                return Ok(default_yes);
            }
            let answer = answer.to_lowercase();
            if answer == "y" || answer == "n" {
                return Ok(answer == "y");
            }
            self.eprintln(Verbosity::Normal, &format!("Did not understand answer '{answer}'"));
        }
    }

    /// `KeyStoreAwareCommand.readPassword`.
    fn read_password(&mut self, verify: bool) -> Result<String, CliError> {
        if !verify {
            return Ok(self.term.read_secret(PROMPT)?);
        }
        let first = self.term.read_secret("Enter new password for the elasticsearch keystore (empty for no password): ")?;
        let second = self.term.read_secret("Enter same password again: ")?;
        if first != second {
            return Err(user(exit::DATA_ERROR, format!("Passwords are not equal, exiting.: {first}, {second}")));
        }
        Ok(first)
    }
}

/// Run the tool with `args` (without the program name) against the config
/// directory `conf`; returns the exit code.
pub fn run(args: &[String], conf: &Path, term: &mut dyn Terminal) -> i32 {
    let mut ctx = Ctx { term, verbosity: Verbosity::Normal };
    let top = match parse(args, &[], true) {
        Ok(o) => o,
        Err(m) => {
            print_help(&mut ctx, None, true);
            ctx.eprintln(Verbosity::Silent, &format!("ERROR: {m}"));
            return exit::USAGE;
        }
    };
    set_verbosity(&mut ctx, &top);
    if top.has('h') {
        print_help(&mut ctx, None, false);
        return exit::OK;
    }
    let Some(cmd) = top.args.first().cloned() else {
        return report(&mut ctx, None, CliError::User(exit::USAGE, Some("Missing required command".into())), true);
    };
    if !COMMANDS.iter().any(|(c, _)| *c == cmd) {
        return report(&mut ctx, None, user(exit::USAGE, format!("Unknown command [{cmd}]")), false);
    }
    let sub = match parse(&top.args[1..], command_options(&cmd), false) {
        Ok(o) => o,
        Err(m) => {
            print_help(&mut ctx, Some(&cmd), true);
            ctx.eprintln(Verbosity::Silent, &format!("ERROR: {m}"));
            return exit::USAGE;
        }
    };
    set_verbosity(&mut ctx, &sub);
    if sub.has('h') {
        print_help(&mut ctx, Some(&cmd), false);
        return exit::OK;
    }
    match execute(&mut ctx, &cmd, &sub, conf) {
        Ok(()) => exit::OK,
        Err(e) => report(&mut ctx, Some(&cmd), e, false),
    }
}

fn set_verbosity(ctx: &mut Ctx<'_>, o: &Opts) {
    if o.has('s') {
        ctx.verbosity = Verbosity::Silent;
    } else if o.has('v') {
        ctx.verbosity = Verbosity::Verbose;
    }
}

fn print_help(ctx: &mut Ctx<'_>, cmd: Option<&str>, to_err: bool) {
    let mut text = String::new();
    match cmd {
        None => {
            text.push_str(DESCRIPTION);
            text.push_str("\n\n");
            if !to_err {
                text.push_str(&command_list());
            }
            text.push_str("Non-option arguments:\ncommand              \n\n");
            text.push_str(&options_help(&[]));
        }
        Some(c) => {
            text.push_str(COMMANDS.iter().find(|(n, _)| *n == c).map(|(_, d)| *d).unwrap_or(""));
            text.push_str("\n\n");
            text.push_str(&options_help(command_options(c)));
        }
    }
    if to_err {
        ctx.eprintln(Verbosity::Silent, &text);
    } else {
        ctx.println(Verbosity::Silent, &text);
    }
}

fn command_list() -> String {
    let mut s = String::from("Commands\n--------\n");
    for (c, d) in COMMANDS {
        s.push_str(&format!("{c} - {d}\n"));
    }
    s.push('\n');
    s
}

/// `Command.main`'s error handling.
fn report(ctx: &mut Ctx<'_>, cmd: Option<&str>, e: CliError, missing_command: bool) -> i32 {
    match e {
        CliError::User(code, msg) => {
            if code == exit::USAGE {
                print_help(ctx, cmd, true);
            }
            if let Some(m) = msg {
                ctx.eprintln(Verbosity::Normal, "");
                ctx.eprintln(Verbosity::Silent, &format!("ERROR: {m}, with exit code {code}"));
            }
            if missing_command {
                ctx.eprintln(Verbosity::Normal, "");
                let list = command_list();
                ctx.eprintln(Verbosity::Normal, list.trim_end_matches('\n'));
            }
            code
        }
        CliError::Keystore(e) => {
            ctx.eprintln(Verbosity::Silent, &format!("{}: {}", e.java_class(), e.message()));
            match e {
                KeystoreError::Corrupt(_) | KeystoreError::Io(_) => exit::IO_ERROR,
                _ => exit::CODE_ERROR,
            }
        }
    }
}

fn execute(ctx: &mut Ctx<'_>, cmd: &str, o: &Opts, conf: &Path) -> Result<(), CliError> {
    match cmd {
        "create" => create(ctx, o, conf),
        "has-passwd" => has_passwd(ctx, conf),
        _ => {
            let must_exist = !matches!(cmd, "add" | "add-file");
            with_keystore(ctx, conf, must_exist, o.has('f'), |ctx, ks, password| match cmd {
                "list" => {
                    for name in ks.setting_names() {
                        ctx.println(Verbosity::Normal, &name);
                    }
                    Ok(())
                }
                "add" => add(ctx, o, conf, ks, password),
                "add-file" => add_file(ctx, o, conf, ks, password),
                "remove" => remove(o, conf, ks, password),
                "show" => show(ctx, o, ks),
                "upgrade" => {
                    ks.upgrade(conf, password)?;
                    Ok(())
                }
                "passwd" => {
                    let new = ctx.read_password(true)?;
                    ks.save(conf, &new)?;
                    ctx.println(Verbosity::Normal, "Elasticsearch keystore password changed successfully.");
                    Ok(())
                }
                other => Err(user(exit::USAGE, format!("Unknown command [{other}]"))),
            })
        }
    }
}

/// `BaseKeyStoreCommand.execute`: load and decrypt (or create) the
/// keystore, run the command; security failures exit with `DATA_ERROR`.
fn with_keystore<F>(ctx: &mut Ctx<'_>, conf: &Path, must_exist: bool, force: bool, f: F) -> Result<(), CliError>
where
    F: FnOnce(&mut Ctx<'_>, &mut KeyStore, &str) -> Result<(), CliError>,
{
    let r = (|| -> Result<(), CliError> {
        let (mut ks, password) = match KeyStore::load(conf)? {
            None => {
                if must_exist {
                    return Err(user(
                        exit::DATA_ERROR,
                        format!("Elasticsearch keystore not found at [{}]. Use 'create' command to create one.", keystore_path(conf).display()),
                    ));
                }
                if !force && !ctx.prompt_yes_no("The elasticsearch keystore does not exist. Do you want to create it?", false)? {
                    ctx.println(Verbosity::Normal, "Exiting without creating keystore.");
                    return Ok(());
                }
                let ks = KeyStore::create()?;
                ks.save(conf, "")?;
                (ks, String::new())
            }
            Some(mut ks) => {
                let password = if ks.has_password() { ctx.read_password(false)? } else { String::new() };
                ks.decrypt(&password)?;
                (ks, password)
            }
        };
        f(ctx, &mut ks, &password)
    })();
    match r {
        Err(CliError::Keystore(e @ (KeystoreError::WrongPassword | KeystoreError::Security(_)))) => Err(user(exit::DATA_ERROR, e.message())),
        other => other,
    }
}

fn create(ctx: &mut Ctx<'_>, o: &Opts, conf: &Path) -> Result<(), CliError> {
    let password = if o.has('p') { ctx.read_password(true)? } else { String::new() };
    let path = keystore_path(conf);
    if path.exists() && !ctx.prompt_yes_no("An elasticsearch keystore already exists. Overwrite?", false)? {
        ctx.println(Verbosity::Normal, "Exiting without creating keystore.");
        return Ok(());
    }
    let r = KeyStore::create().and_then(|ks| ks.save(conf, &password));
    match r {
        Ok(()) => {
            ctx.println(Verbosity::Normal, &format!("Created elasticsearch keystore in {}", path.display()));
            Ok(())
        }
        Err(KeystoreError::WrongPassword | KeystoreError::Security(_)) => Err(user(exit::IO_ERROR, "Error creating the elasticsearch keystore.")),
        Err(e) => Err(e.into()),
    }
}

fn add(ctx: &mut Ctx<'_>, o: &Opts, conf: &Path, ks: &mut KeyStore, password: &str) -> Result<(), CliError> {
    if o.args.is_empty() {
        return Err(user(exit::USAGE, "the setting names can not be empty"));
    }
    for setting in &o.args {
        if ks.contains(setting) && !o.has('f') && !ctx.prompt_yes_no(&format!("Setting {setting} already exists. Overwrite?"), false)? {
            ctx.println(Verbosity::Normal, "Exiting without modifying keystore.");
            return Ok(());
        }
        let prompt = if o.has('x') { String::new() } else { format!("Enter value for {setting}: ") };
        let value = ctx.term.read_secret(&prompt)?;
        if let Err(e) = ks.set_string(setting, &value) {
            return Err(user(exit::DATA_ERROR, e.message()));
        }
    }
    ks.save(conf, password)?;
    Ok(())
}

fn add_file(ctx: &mut Ctx<'_>, o: &Opts, conf: &Path, ks: &mut KeyStore, password: &str) -> Result<(), CliError> {
    if o.args.is_empty() {
        return Err(user(exit::USAGE, "Missing setting name"));
    }
    if o.args.len() % 2 != 0 {
        return Err(user(exit::USAGE, "settings and filenames must come in pairs"));
    }
    for pair in o.args.chunks(2) {
        let setting = &pair[0];
        if ks.contains(setting) && !o.has('f') && !ctx.prompt_yes_no(&format!("Setting {setting} already exists. Overwrite?"), false)? {
            ctx.println(Verbosity::Normal, "Exiting without modifying keystore.");
            return Ok(());
        }
        let file = PathBuf::from(&pair[1]);
        if !file.exists() {
            return Err(user(exit::IO_ERROR, format!("File [{}] does not exist", file.display())));
        }
        let bytes = std::fs::read(&file).map_err(|e| KeystoreError::Io(format!("{}: {e}", file.display())))?;
        ks.set_file(setting, &bytes)?;
    }
    ks.save(conf, password)?;
    Ok(())
}

fn remove(o: &Opts, conf: &Path, ks: &mut KeyStore, password: &str) -> Result<(), CliError> {
    if o.args.is_empty() {
        return Err(user(exit::USAGE, "Must supply at least one setting to remove"));
    }
    for setting in &o.args {
        if !ks.remove(setting) {
            return Err(user(exit::CONFIG, format!("Setting [{setting}] does not exist in the keystore.")));
        }
    }
    ks.save(conf, password)?;
    Ok(())
}

fn show(ctx: &mut Ctx<'_>, o: &Opts, ks: &KeyStore) -> Result<(), CliError> {
    if o.args.len() != 1 {
        return Err(user(exit::USAGE, "Must provide a single setting name to show"));
    }
    let name = &o.args[0];
    let Some(bytes) = ks.get_bytes(name) else {
        return Err(user(exit::CONFIG, format!("Setting [{name}] does not exist in the keystore.")));
    };
    let to_file = ctx.term.output_is_file();
    match std::str::from_utf8(bytes) {
        Ok(text) => {
            if to_file {
                if ctx.verbosity >= Verbosity::Silent {
                    ctx.term.out(text.as_bytes());
                }
            } else {
                ctx.println(Verbosity::Silent, text);
            }
            Ok(())
        }
        Err(e) => {
            if to_file {
                ctx.term.out(bytes);
                return Ok(());
            }
            ctx.eprintln(Verbosity::Verbose, &format!("java.nio.charset.MalformedInputException: {e}"));
            ctx.eprintln(Verbosity::Normal, &format!("The value for the setting [{name}] is not a string and cannot be printed to the console"));
            Err(user(exit::IO_ERROR, "Please redirect binary output to a file instead"))
        }
    }
}

fn has_passwd(ctx: &mut Ctx<'_>, conf: &Path) -> Result<(), CliError> {
    match KeyStore::load(conf)? {
        None => {
            ctx.eprintln(Verbosity::Normal, "ERROR: Elasticsearch keystore not found");
            Err(CliError::User(1, None))
        }
        Some(ks) if !ks.has_password() => {
            ctx.eprintln(Verbosity::Normal, "ERROR: Keystore is not password-protected");
            Err(CliError::User(1, None))
        }
        Some(_) => {
            ctx.println(Verbosity::Normal, "Keystore is password-protected");
            Ok(())
        }
    }
}

/// The `elasticsearch-keystore` binary: `$ES_PATH_CONF` (default `config`)
/// and the process's arguments and standard streams.
pub fn main() -> ! {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let conf = std::env::var_os("ES_PATH_CONF").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("config"));
    let code = run(&args, &conf, &mut SystemTerminal);
    std::process::exit(code)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    #[derive(Default)]
    struct Fake {
        input: VecDeque<String>,
        out: Vec<u8>,
        err: String,
        to_file: bool,
    }

    impl Terminal for Fake {
        fn read_text(&mut self, prompt: &str) -> Result<String, KeystoreError> {
            self.out.extend_from_slice(prompt.as_bytes());
            self.input.pop_front().ok_or_else(|| KeystoreError::IllegalState("unable to read from standard input; is standard input open and a tty attached?".into()))
        }
        fn read_secret(&mut self, prompt: &str) -> Result<String, KeystoreError> {
            self.read_text(prompt)
        }
        fn out(&mut self, text: &[u8]) {
            self.out.extend_from_slice(text);
        }
        fn err(&mut self, text: &str) {
            self.err.push_str(text);
        }
        fn output_is_file(&self) -> bool {
            self.to_file
        }
    }

    fn run_with(conf: &Path, args: &[&str], input: &[&str]) -> (i32, String, String) {
        let mut t = Fake { input: input.iter().map(|s| s.to_string()).collect(), ..Default::default() };
        let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        let code = run(&args, conf, &mut t);
        (code, String::from_utf8_lossy(&t.out).into_owned(), t.err)
    }

    #[test]
    fn create_add_list_show_remove() {
        let dir = tempfile::tempdir().unwrap();
        let conf = dir.path();
        let (code, out, _) = run_with(conf, &["create"], &[]);
        assert_eq!(code, 0);
        assert_eq!(out, format!("Created elasticsearch keystore in {}\n", keystore_path(conf).display()));
        // Overwrite prompt, answered no.
        let (code, out, _) = run_with(conf, &["create"], &["n"]);
        assert_eq!(code, 0);
        assert!(out.ends_with("An elasticsearch keystore already exists. Overwrite? [y/N]Exiting without creating keystore.\n"), "{out}");

        assert_eq!(run_with(conf, &["add", "-x", "s3.client.default.access_key", "s3.client.default.secret_key"], &["AKIA", "SECRET"]).0, 0);
        assert_eq!(run_with(conf, &["add", "bootstrap.password"], &["changeme"]).1, "Enter value for bootstrap.password: ");
        let (code, out, _) = run_with(conf, &["list"], &[]);
        assert_eq!(code, 0);
        assert_eq!(out, "bootstrap.password\nkeystore.seed\ns3.client.default.access_key\ns3.client.default.secret_key\n");
        assert_eq!(run_with(conf, &["show", "s3.client.default.secret_key"], &[]).1, "SECRET\n");

        // Existing setting: prompt; `-f` overwrites without one.
        let (code, out, _) = run_with(conf, &["add", "-x", "bootstrap.password"], &["n"]);
        assert_eq!(code, 0);
        assert!(out.ends_with("Setting bootstrap.password already exists. Overwrite? [y/N]Exiting without modifying keystore.\n"), "{out}");
        assert_eq!(run_with(conf, &["add", "-xf", "bootstrap.password"], &["rotated"]).0, 0);
        assert_eq!(run_with(conf, &["show", "bootstrap.password"], &[]).1, "rotated\n");

        let (code, _, err) = run_with(conf, &["add", "-x", "bad name"], &["v"]);
        assert_eq!(code, 65);
        assert!(err.contains("ERROR: Setting name [bad name] does not match the allowed setting name pattern [[A-Za-z0-9_\\-.]+], with exit code 65"), "{err}");
        let (code, _, err) = run_with(conf, &["add"], &[]);
        assert_eq!(code, 64);
        assert!(err.ends_with("ERROR: the setting names can not be empty, with exit code 64\n"), "{err}");

        assert_eq!(run_with(conf, &["remove", "s3.client.default.secret_key"], &[]).0, 0);
        let (code, _, err) = run_with(conf, &["remove", "s3.client.default.secret_key"], &[]);
        assert_eq!(code, 78);
        assert!(err.ends_with("ERROR: Setting [s3.client.default.secret_key] does not exist in the keystore., with exit code 78\n"), "{err}");
        let (code, _, err) = run_with(conf, &["show", "nope"], &[]);
        assert_eq!(code, 78);
        assert!(err.contains("Setting [nope] does not exist in the keystore."));
        assert_eq!(run_with(conf, &["show"], &[]).0, 64);

        // The node reads what the tool wrote.
        let mut ks = KeyStore::load(conf).unwrap().unwrap();
        ks.decrypt("").unwrap();
        assert_eq!(ks.get_string("s3.client.default.access_key").as_deref(), Some("AKIA"));
        assert!(!ks.contains("s3.client.default.secret_key"));
    }

    #[test]
    fn add_file_and_binary_show() {
        let dir = tempfile::tempdir().unwrap();
        let conf = dir.path();
        let file = conf.join("creds.bin");
        std::fs::write(&file, [0xffu8, 0x00, 0x01]).unwrap();
        // Missing keystore: `-f` creates it without asking.
        assert_eq!(run_with(conf, &["add-file", "-f", "gcs.client.default.credentials_file", file.to_str().unwrap()], &[]).0, 0);
        let (code, _, err) = run_with(conf, &["show", "gcs.client.default.credentials_file"], &[]);
        assert_eq!(code, 74);
        assert!(err.contains("The value for the setting [gcs.client.default.credentials_file] is not a string and cannot be printed to the console"));
        assert!(err.ends_with("ERROR: Please redirect binary output to a file instead, with exit code 74\n"), "{err}");
        let mut t = Fake { to_file: true, ..Default::default() };
        assert_eq!(run(&["show".into(), "gcs.client.default.credentials_file".into()], conf, &mut t), 0);
        assert_eq!(t.out, vec![0xff, 0x00, 0x01]);
        let (code, _, err) = run_with(conf, &["add-file", "x.y", "/definitely/missing"], &[]);
        assert_eq!(code, 74);
        assert!(err.contains("ERROR: File [/definitely/missing] does not exist, with exit code 74"));
        assert_eq!(run_with(conf, &["add-file", "x.y"], &[]).0, 64);
    }

    #[test]
    fn missing_keystore_and_prompts() {
        let dir = tempfile::tempdir().unwrap();
        let conf = dir.path();
        let (code, _, err) = run_with(conf, &["list"], &[]);
        assert_eq!(code, 65);
        assert!(err.ends_with(&format!(
            "ERROR: Elasticsearch keystore not found at [{}]. Use 'create' command to create one., with exit code 65\n",
            keystore_path(conf).display()
        )));
        let (code, out, _) = run_with(conf, &["add", "a.b"], &["n"]);
        assert_eq!(code, 0);
        assert_eq!(out, "The elasticsearch keystore does not exist. Do you want to create it? [y/N]Exiting without creating keystore.\n");
        assert!(!keystore_path(conf).exists());
        let (code, _, err) = run_with(conf, &["has-passwd"], &[]);
        assert_eq!((code, err.as_str()), (1, "ERROR: Elasticsearch keystore not found\n"));
        assert_eq!(run_with(conf, &["has-passwd", "--silent"], &[]), (1, String::new(), String::new()));
        // Unknown command / option, no command.
        let (code, _, err) = run_with(conf, &["frobnicate"], &[]);
        assert_eq!(code, 64);
        assert!(err.ends_with("ERROR: Unknown command [frobnicate], with exit code 64\n"), "{err}");
        let (code, _, err) = run_with(conf, &["list", "--bogus"], &[]);
        assert_eq!(code, 64);
        assert!(err.ends_with("ERROR: bogus is not a recognized option\n"), "{err}");
        let (code, _, err) = run_with(conf, &[], &[]);
        assert_eq!(code, 64);
        assert!(err.contains("ERROR: Missing required command, with exit code 64"));
        assert!(err.contains("has-passwd - Succeeds if the keystore exists"));
    }

    #[test]
    fn password_protected_keystore() {
        let dir = tempfile::tempdir().unwrap();
        let conf = dir.path();
        let (code, _, err) = run_with(conf, &["create", "-p"], &["one", "two"]);
        assert_eq!(code, 65);
        assert!(err.contains("ERROR: Passwords are not equal, exiting.: one, two, with exit code 65"));
        assert_eq!(run_with(conf, &["create", "-p"], &["s3cret", "s3cret"]).0, 0);
        let (code, out, _) = run_with(conf, &["has-passwd"], &[]);
        assert_eq!((code, out.as_str()), (0, "Keystore is password-protected\n"));
        // Password first, then the value.
        assert_eq!(run_with(conf, &["add", "-x", "bootstrap.password"], &["s3cret", "changeme"]).0, 0);
        let (code, _, err) = run_with(conf, &["list"], &["wrong"]);
        assert_eq!(code, 65);
        assert!(err.ends_with("ERROR: Provided keystore password was incorrect, with exit code 65\n"), "{err}");
        // passwd: current password, then the new one twice; empty removes it.
        let (code, out, _) = run_with(conf, &["passwd"], &["s3cret", "", ""]);
        assert_eq!(code, 0);
        assert!(out.ends_with("Elasticsearch keystore password changed successfully.\n"));
        let (code, _, err) = run_with(conf, &["has-passwd"], &[]);
        assert_eq!((code, err.as_str()), (1, "ERROR: Keystore is not password-protected\n"));
        assert_eq!(run_with(conf, &["show", "bootstrap.password"], &[]).1, "changeme\n");
    }

    #[test]
    fn upgrade_and_corruption() {
        let dir = tempfile::tempdir().unwrap();
        let conf = dir.path();
        std::fs::write(keystore_path(conf), include_bytes!("testdata/format-v4-elasticsearch.keystore")).unwrap();
        assert_eq!(run_with(conf, &["upgrade"], &[]).0, 0);
        assert_eq!(KeyStore::load(conf).unwrap().unwrap().format_version(), super::super::CURRENT_VERSION);
        let mut bytes = std::fs::read(keystore_path(conf)).unwrap();
        bytes[50] ^= 1;
        std::fs::write(keystore_path(conf), &bytes).unwrap();
        let (code, _, err) = run_with(conf, &["list"], &[]);
        assert_eq!(code, 74);
        assert!(err.starts_with("org.apache.lucene.index.CorruptIndexException: checksum failed (hardware problem?)"), "{err}");
    }
}
