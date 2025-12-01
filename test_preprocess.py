from preprocess import preprocess_puzzle_image, save_pieces


def main():
    image_path = "test_cases/sample1/sample1_translate.png"

    # 2. preprocess
    # --- 修复：这里需要用两个变量接收返回值 (pieces 和 config) ---
    pieces, config = preprocess_puzzle_image(image_path, None, None, debug=True)

    # 打印一下检测结果看看
    print(f"Auto-Detected Config: {config}")
    print(f"detect {len(pieces)} pieces in total")

    # 3. check each piece
    out_dir = image_path[:-4] + "_preprocess1"

    # save_pieces 只接受列表，现在 pieces 已经是正确的列表了
    save_pieces(pieces, out_dir, save_meta=True)
    print(f"saved to dir: {out_dir}")

    # 4. print the first piece edge feature
    if pieces:
        p0 = pieces[0]
        print(f"piece zero: {p0.size}, corners: \n{p0.canvas_corners}")
        for edge_name, feat in p0.edges.items():
            print(f"  edge {edge_name}: mean_color={feat.mean_color}, "
                  f"hist_len={len(feat.color_hist)}")


if __name__ == "__main__":
    main()