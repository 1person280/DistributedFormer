# -*- coding: utf-8 -*-
"""
Rust Coding 真实需求基准数据集 (v1)

数据来源: 真实 Rust 开发中的高频痛点。每条样本 = 一段真实风格的 Rust
代码片段 + 其真实 rustc 错误类别 (E 编码) + rustc 实际报错信息。
类别对应 Rust 开发者日常遇到最多的编译错误族:

  move      所有权移动错误     E0382 / E0505 / E0507
  borrow    借用冲突           E0502 / E0499
  lifetime  生命周期错误       E0597 / E0106 / E0515 / E0716 / E0623
  type      类型不匹配         E0308 / E0277 / E0599 / E0300
  ok        可正常编译的对照代码

设计动机 (以真实需求为主): 框架此前的训练监督全部来自合成随机数据,
存在"知识脱离实际"的问题。本数据集让模型直接消费真实代码文本,
面向的真实需求是: 静态识别一段 Rust 代码命中哪一类编译错误 ——
这是 lints / IDE 提示 / 自动修复工具的基础能力。

用法:
    from distributedformer.data.rust_coding import (
        load_rust_coding, static_metrics, stratified_split
    )
"""

import numpy as np

LABELS = ["move", "borrow", "lifetime", "type", "ok"]
LABEL_NAMES = {
    "move": "所有权移动 (E0382/E0505/E0507)",
    "borrow": "借用冲突 (E0502/E0499)",
    "lifetime": "生命周期 (E0597/E0106/E0515/E0716)",
    "type": "类型不匹配 (E0308/E0277/E0599)",
    "ok": "合法代码 (可编译)",
}

# 每条: code=代码片段, label=类别, rustc=rustc 错误码, msg=真实报错信息摘要
RUST_SNIPPETS = [
    # ── move: 所有权移动 ──────────────────────────────────────
    {"code": 'let s = String::from("hi");\nlet t = s;\nprintln!("{}", s);',
     "label": "move", "rustc": "E0382", "msg": "use of moved value: `s`"},
    {"code": 'let v = vec![1, 2];\nlet w = v;\nprintln!("{:?}", v);',
     "label": "move", "rustc": "E0382", "msg": "use of moved value: `v`"},
    {"code": 'fn take(s: String) {}\nlet s = String::from("x");\ntake(s);\nprintln!("{}", s);',
     "label": "move", "rustc": "E0382", "msg": "use of moved value: `s`"},
    {"code": 'let a = String::from("a");\nlet b = a;\nlet c = a;',
     "label": "move", "rustc": "E0382", "msg": "use of moved value: `a`"},
    {"code": 'struct P { name: String }\nlet p = P { name: String::from("n") };\nlet q = p;\nprintln!("{}", p.name);',
     "label": "move", "rustc": "E0382", "msg": "use of moved value: `p`"},
    {"code": 'let s = String::from("m");\nlet u = s;\ndrop(s);',
     "label": "move", "rustc": "E0382", "msg": "use of moved value: `s`"},
    {"code": 'let v = vec![String::from("a")];\nlet first = v[0];\nprintln!("{:?}", v);',
     "label": "move", "rustc": "E0507", "msg": "cannot move out of index of `Vec<String>`"},
    {"code": 'let s = String::from("z");\nfor c in s {}\nprintln!("{}", s);',
     "label": "move", "rustc": "E0382", "msg": "use of moved value: `s`"},
    {"code": "let x = Box::new(5);\nlet y = x;\nprintln!(\"{}\", x);",
     "label": "move", "rustc": "E0382", "msg": "use of moved value: `x`"},
    {"code": 'fn consume(v: Vec<i32>) {}\nlet v = vec![1];\nconsume(v);\nconsume(v);',
     "label": "move", "rustc": "E0382", "msg": "use of moved value: `v`"},
    {"code": 'let tuple = (String::from("a"), 1);\nlet (s, _) = tuple;\nlet (s2, _) = tuple;',
     "label": "move", "rustc": "E0382", "msg": "use of moved value: `tuple`"},
    {"code": 'let s = String::from("t");\nlet r = &s;\nlet o = s;\nprintln!("{}", r);',
     "label": "move", "rustc": "E0505", "msg": "cannot move out of `s` because it is borrowed"},
    {"code": 'let d = String::from("d");\nlet f = move || println!("{}", d);\nf();\nf();',
     "label": "move", "rustc": "E0382", "msg": "use of moved value in closure"},
    {"code": 'let m = std::collections::HashMap::new();\nm.insert("k", String::from("v"));\nlet val = m["k"];',
     "label": "move", "rustc": "E0507", "msg": "cannot move out of index of `HashMap`"},
    {"code": 'let s = String::from("q");\nlet arr = [s, s];',
     "label": "move", "rustc": "E0382", "msg": "use of moved value: `s`"},
    {"code": 'let b = Box::new(String::from("b"));\nlet c = *b;\nprintln!("{}", b);',
     "label": "move", "rustc": "E0507", "msg": "cannot move out of `*b`"},
    {"code": 'let s = String::from("w");\nlet t = s;\ns.push_str("x");',
     "label": "move", "rustc": "E0382", "msg": "use of moved value: `s`"},
    {"code": "let v = vec![1, 2, 3];\nlet it = v.into_iter();\nlet _ = v.len();",
     "label": "move", "rustc": "E0382", "msg": "use of moved value: `v`"},
    {"code": 'fn id(s: String) -> String { s }\nlet s = String::from("i");\nlet t = id(s);\nlet u = id(s);',
     "label": "move", "rustc": "E0382", "msg": "use of moved value: `s`"},
    {"code": 'let s1 = String::from("s1");\nlet s2 = String::from("s2");\nlet v = vec![s1, s2];\nlet out = format!("{}{}", s1, s2);',
     "label": "move", "rustc": "E0382", "msg": "use of moved value: `s1`"},

    # ── borrow: 借用冲突 ──────────────────────────────────────
    {"code": 'let mut v = vec![1];\nlet r = &v;\nv.push(2);\nprintln!("{:?}", r);',
     "label": "borrow", "rustc": "E0502", "msg": "cannot borrow `v` as mutable because it is also borrowed as immutable"},
    {"code": 'let mut s = String::from("a");\nlet r1 = &s;\nlet r2 = &mut s;\nprintln!("{}{}", r1, r2);',
     "label": "borrow", "rustc": "E0502", "msg": "cannot borrow `s` as mutable because it is also borrowed as immutable"},
    {"code": "let mut x = 5;\nlet r = &x;\nx += 1;\nprintln!(\"{}\", r);",
     "label": "borrow", "rustc": "E0502", "msg": "cannot borrow `x` as mutable because it is also borrowed as immutable"},
    {"code": 'let mut v = vec![1, 2];\nlet first = &v[0];\nv.push(3);\nprintln!("{}", first);',
     "label": "borrow", "rustc": "E0502", "msg": "cannot borrow `v` as mutable because it is also borrowed as immutable"},
    {"code": 'let mut s = String::from("b");\nlet r = &mut s;\nlet r2 = &mut s;\nprintln!("{}{}", r, r2);',
     "label": "borrow", "rustc": "E0499", "msg": "cannot borrow `s` as mutable more than once at a time"},
    {"code": 'fn push(v: &mut Vec<i32>) {}\nlet mut v = vec![];\nlet r = &v;\npush(&mut v);',
     "label": "borrow", "rustc": "E0502", "msg": "cannot borrow `v` as mutable because it is also borrowed as immutable"},
    {"code": "let mut m = std::collections::HashMap::new();\nm.insert(1, 2);\nlet k = m.keys();\nm.insert(3, 4);\nfor _x in k {}",
     "label": "borrow", "rustc": "E0502", "msg": "cannot borrow `m` as mutable because it is also borrowed as immutable"},
    {"code": 'let mut s = String::new();\nlet r = &s;\ns.push(\'a\');\nprintln!("{}", r);',
     "label": "borrow", "rustc": "E0502", "msg": "cannot borrow `s` as mutable because it is also borrowed as immutable"},
    {"code": "let mut a = [1, 2, 3];\nlet r = &a[..];\na[0] = 9;\nprintln!(\"{:?}\", r);",
     "label": "borrow", "rustc": "E0502", "msg": "cannot borrow `a` as mutable because it is also borrowed as immutable"},
    {"code": "let mut v = vec![1, 2];\nlet i1 = v.iter_mut();\nlet i2 = v.iter_mut();",
     "label": "borrow", "rustc": "E0499", "msg": "cannot borrow `v` as mutable more than once at a time"},
    {"code": 'let mut s = String::from("c");\nlet r1 = &s;\nlet r2 = &s;\ns.push(\'x\');\nprintln!("{}{}", r1, r2);',
     "label": "borrow", "rustc": "E0502", "msg": "cannot borrow `s` as mutable because it is also borrowed as immutable"},
    {"code": "struct H { n: i32 }\nlet mut h = H { n: 1 };\nlet rn = &h.n;\nh.n = 2;\nprintln!(\"{}\", rn);",
     "label": "borrow", "rustc": "E0502", "msg": "cannot borrow `h.n` as mutable"},
    {"code": 'let mut v = vec![1, 2];\nlet r = &v;\nv.clear();\nprintln!("{:?}", r);',
     "label": "borrow", "rustc": "E0502", "msg": "cannot borrow `v` as mutable because it is also borrowed as immutable"},
    {"code": 'fn get(s: &String) -> &String { s }\nlet mut s = String::from("d");\nlet g = get(&s);\ns.push(\'!\');\nprintln!("{}", g);',
     "label": "borrow", "rustc": "E0502", "msg": "cannot borrow `s` as mutable because it is also borrowed as immutable"},
    {"code": 'let mut v: Vec<i32> = Vec::new();\nlet r = &v;\nv.push(1);\nprintln!("{:?}", r);',
     "label": "borrow", "rustc": "E0502", "msg": "cannot borrow `v` as mutable because it is also borrowed as immutable"},
    {"code": "let mut x = Box::new(3);\nlet r = &*x;\n*x = 4;\nprintln!(\"{}\", r);",
     "label": "borrow", "rustc": "E0502", "msg": "cannot borrow `*x` as immutable because it is also borrowed as mutable"},
    {"code": 'let mut s = String::from("e");\nlet p = &s.len();\nlet q = &mut s;\nprintln!("{}{}", p, q);',
     "label": "borrow", "rustc": "E0502", "msg": "cannot borrow `s` as mutable because it is also borrowed as immutable"},
    {"code": 'let mut v = vec![1, 2, 3];\nlet slice = &v[1..];\nv[1] = 0;\nprintln!("{:?}", slice);',
     "label": "borrow", "rustc": "E0502", "msg": "cannot borrow `v` as mutable because it is also borrowed as immutable"},
    {"code": 'let mut m = std::collections::HashMap::new();\nm.insert("a", 1);\nlet r = m.get("a");\nm.insert("b", 2);\nprintln!("{:?}", r);',
     "label": "borrow", "rustc": "E0502", "msg": "cannot borrow `m` as mutable because it is also borrowed as immutable"},
    {"code": "let mut n = 10;\nlet r = &n;\nlet m2 = &mut n;\nprintln!(\"{}{}\", r, m2);",
     "label": "borrow", "rustc": "E0502", "msg": "cannot borrow `n` as mutable because it is also borrowed as immutable"},

    # ── lifetime: 生命周期 ────────────────────────────────────
    {"code": "fn r() -> &i32 {\n    let x = 5;\n    &x\n}",
     "label": "lifetime", "rustc": "E0597", "msg": "`x` does not live long enough"},
    {"code": "fn longest(s1: &str, s2: &str) -> &str {\n    if s1.len() > s2.len() { s1 } else { s2 }\n}",
     "label": "lifetime", "rustc": "E0106", "msg": "missing lifetime specifier"},
    {"code": "let r;\n{\n    let x = 5;\n    r = &x;\n}\nprintln!(\"{}\", r);",
     "label": "lifetime", "rustc": "E0597", "msg": "`x` does not live long enough"},
    {"code": 'let d;\n{\n    let s = String::from("l");\n    d = &s;\n}\nprintln!("{}", d);',
     "label": "lifetime", "rustc": "E0597", "msg": "`s` does not live long enough"},
    {"code": 'fn f() -> &String {\n    &String::from("t")\n}',
     "label": "lifetime", "rustc": "E0716", "msg": "temporary value dropped while borrowed"},
    {"code": "struct Holder<'a> { s: &'a str }\nlet h;\n{\n    let s = String::from(\"h\");\n    h = Holder { s: &s };\n}\nprintln!(\"{}\", h.s);",
     "label": "lifetime", "rustc": "E0597", "msg": "`s` does not live long enough"},
    {"code": "fn f<'a>() -> &'a i32 {\n    let x = 1;\n    &x\n}",
     "label": "lifetime", "rustc": "E0597", "msg": "`x` does not live long enough"},
    {"code": "struct S { r: &i32 }",
     "label": "lifetime", "rustc": "E0106", "msg": "missing lifetime specifier"},
    {"code": "fn pick(a: &Vec<i32>, b: &Vec<i32>) -> &i32 {\n    if a[0] > b[0] { &a[0] } else { &b[0] }\n}",
     "label": "lifetime", "rustc": "E0106", "msg": "missing lifetime specifier"},
    {"code": 'let v = vec![1, 2];\nlet r = &v[0];\ndrop(v);\nprintln!("{}", r);',
     "label": "lifetime", "rustc": "E0597", "msg": "`v` does not live long enough"},
    {"code": "fn f(x: &i32) -> &i32 {\n    let y = 0;\n    if *x > 0 { &y } else { x }\n}",
     "label": "lifetime", "rustc": "E0515", "msg": "cannot return reference to local variable `y`"},
    {"code": "fn f() -> &i32 {\n    &42\n}",
     "label": "lifetime", "rustc": "E0515", "msg": "cannot return reference to temporary value"},
    {"code": "let r = {\n    let x = 3;\n    &x\n};",
     "label": "lifetime", "rustc": "E0597", "msg": "`x` does not live long enough"},
    {"code": "fn dangle() -> &str {\n    let s = String::from(\"d\");\n    &s\n}",
     "label": "lifetime", "rustc": "E0515", "msg": "cannot return reference to local variable `s`"},
    {"code": 'let r = &String::from("mk");\nprintln!("{}", r);',
     "label": "lifetime", "rustc": "E0716", "msg": "temporary value dropped while borrowed"},
    {"code": "fn f<'a>(x: &'a str, y: &str, c: bool) -> &'a str {\n    if c { x } else { y }\n}",
     "label": "lifetime", "rustc": "E0623", "msg": "lifetime mismatch"},
    {"code": 'fn f() -> &\'static str {\n    let s = String::from("s");\n    &s\n}',
     "label": "lifetime", "rustc": "E0597", "msg": "`s` does not live long enough"},
    {"code": 'let long = String::from("long");\nlet r;\n{\n    let short = String::from("s");\n    r = if long.len() > 1 { &long } else { &short };\n}\nprintln!("{}", r);',
     "label": "lifetime", "rustc": "E0597", "msg": "`short` does not live long enough"},
    {"code": "fn f() -> &[i32] {\n    let v = vec![1, 2];\n    &v\n}",
     "label": "lifetime", "rustc": "E0515", "msg": "cannot return reference to local variable `v`"},
    {"code": "fn f<'a>(x: &'a i32) -> &i32 {\n    x\n}",
     "label": "lifetime", "rustc": "E0106", "msg": "missing lifetime specifier"},

    # ── type: 类型不匹配 ──────────────────────────────────────
    {"code": 'let x: i32 = "hello";',
     "label": "type", "rustc": "E0308", "msg": "mismatched types: expected `i32`, found `&str`"},
    {"code": "let v: Vec<i32> = vec![1.0, 2.0];",
     "label": "type", "rustc": "E0308", "msg": "mismatched types: expected `i32`, found floating-point"},
    {"code": 'fn f(x: i32) {}\nf("s");',
     "label": "type", "rustc": "E0308", "msg": "mismatched types: expected `i32`, found `&str`"},
    {"code": "let s: String = 42;",
     "label": "type", "rustc": "E0308", "msg": "mismatched types: expected `String`, found integer"},
    {"code": "let b: bool = 1;",
     "label": "type", "rustc": "E0308", "msg": "mismatched types: expected `bool`, found integer"},
    {"code": 'let t: (i32, &str) = ("a", 1);',
     "label": "type", "rustc": "E0308", "msg": "mismatched types"},
    {"code": 'let x = 5;\nx.foo();',
     "label": "type", "rustc": "E0599", "msg": "no method named `foo` found"},
    {"code": "struct S;\nS::new();",
     "label": "type", "rustc": "E0599", "msg": "no function or associated item named `new` found"},
    {"code": "fn p<T: std::fmt::Display>(t: T) {}\np(vec![1]);",
     "label": "type", "rustc": "E0277", "msg": "`Vec<i32>` doesn't implement `std::fmt::Display`"},
    {"code": "fn p<T: Clone>(t: T) {}\nstruct S;\np(S);",
     "label": "type", "rustc": "E0277", "msg": "`S` doesn't implement `Clone`"},
    {"code": "let x: u8 = 300;",
     "label": "type", "rustc": "E0300", "msg": "literal out of range for `u8`"},
    {"code": 'let mut v = vec![1, 2];\nv.push("3");',
     "label": "type", "rustc": "E0308", "msg": "mismatched types: expected `i32`, found `&str`"},
    {"code": "let a: i32 = 1.5;",
     "label": "type", "rustc": "E0308", "msg": "mismatched types: expected `i32`, found floating-point"},
    {"code": 'fn f() -> i32 {\n    "text"\n}',
     "label": "type", "rustc": "E0308", "msg": "mismatched types: expected `i32`, found `&str`"},
    {"code": "let arr: [i32; 3] = [1, 2];",
     "label": "type", "rustc": "E0308", "msg": "mismatched types: expected an array with a fixed size of 3"},
    {"code": 'let s = "abc";\nlet n: u32 = s;',
     "label": "type", "rustc": "E0308", "msg": "mismatched types: expected `u32`, found `&str`"},
    {"code": 'let opt: Option<i32> = Some("x");',
     "label": "type", "rustc": "E0308", "msg": "mismatched types: expected `i32`, found `&str`"},
    {"code": "trait G { fn g(&self); }\nstruct S;\nimpl G for S { fn g(&self) {} }\nlet x: dyn G = S;",
     "label": "type", "rustc": "E0277", "msg": "the size for values of type `S` cannot be known at compilation time"},
    {"code": 'fn f(x: String) {}\nf(&String::from("y"));',
     "label": "type", "rustc": "E0308", "msg": "mismatched types: expected `String`, found `&String`"},
    {"code": 'let v: Vec<Box<dyn std::fmt::Debug>> = vec![5];',
     "label": "type", "rustc": "E0308", "msg": "mismatched types: expected `Box<dyn Debug>`, found integer"},

    # ── ok: 合法代码 ──────────────────────────────────────────
    {"code": 'let s = String::from("ok");\nprintln!("{}", s);',
     "label": "ok", "rustc": "-", "msg": "compiles"},
    {"code": 'let v = vec![1, 2, 3];\nfor x in &v { println!("{}", x); }',
     "label": "ok", "rustc": "-", "msg": "compiles"},
    {"code": 'let mut v = vec![1];\nv.push(2);\nprintln!("{:?}", v);',
     "label": "ok", "rustc": "-", "msg": "compiles"},
    {"code": 'let v = vec![1];\nlet r = &v;\nprintln!("{:?}", r);',
     "label": "ok", "rustc": "-", "msg": "compiles"},
    {"code": 'let mut s = String::from("a");\nlet r = &s;\nprintln!("{} {}", r, s);',
     "label": "ok", "rustc": "-", "msg": "compiles"},
    {"code": "let mut x = 5;\n{\n    let r = &mut x;\n    *r += 1;\n}\nprintln!(\"{}\", x);",
     "label": "ok", "rustc": "-", "msg": "compiles"},
    {"code": 'let s = String::from("clone");\nlet t = s.clone();\nprintln!("{} {}", s, t);',
     "label": "ok", "rustc": "-", "msg": "compiles"},
    {"code": "let a = [1, 2, 3];\nlet slice = &a[1..];\nprintln!(\"{:?}\", slice);",
     "label": "ok", "rustc": "-", "msg": "compiles"},
    {"code": 'let mut m = std::collections::HashMap::new();\nm.insert("k", 1);\nprintln!("{:?}", m);',
     "label": "ok", "rustc": "-", "msg": "compiles"},
    {"code": 'fn add(a: i32, b: i32) -> i32 { a + b }\nprintln!("{}", add(1, 2));',
     "label": "ok", "rustc": "-", "msg": "compiles"},
    {"code": 'let t = (1, "two");\nprintln!("{} {}", t.0, t.1);',
     "label": "ok", "rustc": "-", "msg": "compiles"},
    {"code": "let opt = Some(5);\nif let Some(n) = opt { println!(\"{}\", n); }",
     "label": "ok", "rustc": "-", "msg": "compiles"},
    {"code": 'let s = "static";\nlet r: &str = s;\nprintln!("{}", r);',
     "label": "ok", "rustc": "-", "msg": "compiles"},
    {"code": 'let v: Vec<String> = vec!["a".to_string(), "b".to_string()];',
     "label": "ok", "rustc": "-", "msg": "compiles"},
    {"code": "let mut n = 0;\nwhile n < 3 { n += 1; }\nprintln!(\"{}\", n);",
     "label": "ok", "rustc": "-", "msg": "compiles"},
    {"code": 'let f = |x: i32| x * 2;\nprintln!("{}", f(21));',
     "label": "ok", "rustc": "-", "msg": "compiles"},
    {"code": 'let s1 = String::from("x");\nlet s2 = String::from("y");\nlet c = format!("{}{}", s1, s2);\nprintln!("{}", c);',
     "label": "ok", "rustc": "-", "msg": "compiles"},
    {"code": 'let mut v = vec![3, 1, 2];\nv.sort();\nprintln!("{:?}", v);',
     "label": "ok", "rustc": "-", "msg": "compiles"},
    {"code": "let b = Box::new(10);\nprintln!(\"{}\", b);",
     "label": "ok", "rustc": "-", "msg": "compiles"},
    {"code": 'let it = vec![1, 2].iter().map(|x| x + 1).collect::<Vec<_>>();\nprintln!("{:?}", it);',
     "label": "ok", "rustc": "-", "msg": "compiles"},
]


def load_rust_coding() -> list:
    """加载数据集: 返回 [{code, label(int), label_name, rustc, msg}, ...]"""
    samples = []
    for s in RUST_SNIPPETS:
        samples.append({
            "code": s["code"],
            "label": LABELS.index(s["label"]),
            "label_name": s["label"],
            "rustc": s["rustc"],
            "msg": s["msg"],
        })
    return samples


def static_metrics(code: str) -> np.ndarray:
    """真实静态分析特征 (模拟 rustc/rustfmt 的轻量扫描):

    [长度, '&'数, 'mut'数, 生命周期符数, 'clone'数, 'move'数,
     'fn'数, '->'数, 嵌套块深度, '::'数]
    """
    return np.array([
        len(code) / 100.0,
        code.count("&"),
        code.count("mut"),
        code.count("'"),
        code.count("clone"),
        code.count("move"),
        code.count("fn"),
        code.count("->"),
        code.count("{") - code.count("}"),
        code.count("::"),
    ], dtype=float)


def structure_metrics(code: str) -> np.ndarray:
    """结构感知特征 (P1, v0.8.x): 针对 v0.7.5 诊断的混淆源设计

    [ '&mut'数, 返回引用'-> &'数, 类型标注':'数, 'println'数,
      'let'绑定数, 防御调用数(clone+to_string+copy) ]

    设计依据 (5 种子混淆矩阵):
    - move↔lifetime 互混: 返回引用位置 (-> &) 是 lifetime 强信号,
      use-after-move 常伴随 println 使用已移动变量
    - type→ok 误判: 类型标注 ': ' 密度区分显式类型代码
    - ok 类防御性调用 (clone/to_string) 显著多于错误类
    """
    return np.array([
        code.count("&mut"),
        code.count("-> &") + code.count("->&") + code.count("-> &'")
        + code.count("->&'"),
        code.count(": "),
        code.count("println"),
        code.count("let "),
        code.count("clone") + code.count("to_string") + code.count(".copy"),
    ], dtype=float)


def stratified_split(samples: list, train_ratio: float = 0.75,
                     seed: int = 0):
    """分层划分训练/验证集 (每类别按比例抽取)"""
    rng = np.random.RandomState(seed)
    by_label = {}
    for s in samples:
        by_label.setdefault(s["label"], []).append(s)
    train, val = [], []
    for label, items in sorted(by_label.items()):
        idx = rng.permutation(len(items))
        n_train = max(1, int(round(len(items) * train_ratio)))
        train.extend(items[i] for i in idx[:n_train])
        val.extend(items[i] for i in idx[n_train:])
    return train, val


def stratified_kfold(samples: list, n_folds: int = 5, seed: int = 0):
    """分层 K 折划分 (每类别轮流分配到各折, 折间类别比例一致)

    Returns:
        folds: 长度 n_folds 的列表, 每项为一折的验证样本;
        训练集 = 其余折的并集。替代单次 75/25 划分, 评估结论
        不再依赖划分运气 (P2 交叉验证)。
    """
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(n_folds)]
    by_label = {}
    for s in samples:
        by_label.setdefault(s["label"], []).append(s)
    for label, items in sorted(by_label.items()):
        idx = rng.permutation(len(items))
        for pos, i in enumerate(idx):
            folds[pos % n_folds].append(items[i])
    return folds
