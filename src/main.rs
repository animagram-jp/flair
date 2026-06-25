// SPDX-License-Identifier: Apache-2.0

mod svd;

fn main() {
    // 動作確認: 3×2 行列の SVD
    let a = vec![
        vec![1.0_f64, 2.0],
        vec![3.0, 4.0],
        vec![5.0, 6.0],
    ];
    let (u, s, vt) = svd::svd(&a);
    println!("s = {:?}", s);
    println!("U rows = {}, cols = {}", u.len(), u.first().map_or(0, |r| r.len()));
    println!("Vt rows = {}, cols = {}", vt.len(), vt.first().map_or(0, |r| r.len()));

    // svdvals のみ
    let s2 = svd::svdvals(&a);
    println!("svdvals = {:?}", s2);
}
