from preprocess import preprocess_puzzle_image, save_pieces


def main():
    image_path = "starry_night_translate.rgb"

    # 2. preprocess
    pieces = preprocess_puzzle_image(image_path, 800, 800, debug=True)
    print(f"detect {len(pieces)} pieces in total")

    # 3. check each piece
    out_dir = image_path[:-4] + "_preprocess"
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
