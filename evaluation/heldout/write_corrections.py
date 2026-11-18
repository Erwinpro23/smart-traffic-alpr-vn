"""Ghi corrections.json từ kết quả duyệt bằng mắt (Claude, AI) các ảnh review2 và bảng crop xe buýt/xe tải.
Quy ước: xe van chở khách/chở hàng liền khối -> car; xe tải thùng có cabin riêng -> truck; xe khách 16 chỗ -> bus;
tủ điện, nhà chờ, biển hiệu, cụm người đi xe máy bị gọi bus/truck -> bỏ; xe ba gác (không thuộc 4 lớp) -> bỏ.
Dùng: python -m evaluation.heldout.write_corrections"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

HEM = {
    "hem_01451.jpg": {"drop": [3]},        # box thứ hai chỉ phủ nửa dưới cùng một xe (áo mưa đỏ)
    "hem_01141.jpg": {"drop": [2]},        # dải hẹp ở mép trái, không xác định được là xe
    "hem_01219.jpg": {"drop": [1]},        # trùng với box 3 (cùng xe máy chở hàng)
    "hem_01257.jpg": {"drop": [2, 4]},     # 2: "car" trên người lái xe máy; 4: trùng box 1
}

# junction: (frame, index) -> lớp mới, hoặc None = bỏ
J = {
    ("00039", 7): "truck", ("00119", 4): "truck", ("00199", 16): "car", ("00199", 23): None, ("00199", 24): None,
    ("00199", 27): None, ("00279", 20): None, ("00359", 20): None, ("00439", 10): None, ("00439", 14): None,
    ("00519", 1): "car", ("00519", 8): None, ("00519", 20): None, ("00599", 5): "car", ("00599", 18): None,
    ("00599", 22): None, ("00678", 6): "car", ("00678", 19): None, ("00678", 24): None, ("00758", 10): "car",
    ("00758", 16): "car", ("00838", 10): "car", ("00838", 11): None, ("00838", 24): None, ("00918", 22): None,
    ("00918", 27): None, ("00998", 3): None, ("01078", 14): None, ("01158", 12): None, ("01158", 13): None,
    ("01238", 11): None, ("01238", 13): None, ("01238", 24): None, ("01238", 27): None, ("01317", 14): None,
    ("01317", 21): None, ("01397", 16): None, ("01477", 4): None, ("01477", 24): None, ("01557", 16): None,
    ("01557", 24): None, ("01637", 10): None, ("01637", 11): None, ("01637", 25): None, ("01717", 17): None,
    ("01717", 22): None, ("01717", 27): None, ("01797", 17): None, ("01797", 18): None, ("01797", 25): None,
    ("01877", 5): "car", ("01877", 22): "car", ("01877", 25): None, ("01956", 2): "car", ("01956", 10): None,
    ("01956", 23): None, ("01956", 25): None, ("02036", 21): None, ("02116", 1): "car", ("02116", 2): "car",
    ("02116", 6): "car", ("02116", 23): None, ("02196", 8): "car", ("02196", 13): None, ("02196", 14): "car",
    ("02196", 25): None, ("02196", 26): None, ("02276", 10): None, ("02276", 12): None, ("02276", 25): None,
    ("02356", 16): None, ("02436", 7): "car", ("02436", 10): None, ("02436", 15): None, ("02516", 2): None,
    ("02595", 11): "car", ("02675", 9): "car", ("02675", 16): None, ("02755", 9): "car", ("02755", 22): "car",
    ("02835", 12): "car", ("02835", 13): None, ("02835", 21): None, ("02835", 24): "car", ("02835", 27): "car",
    ("02915", 2): None, ("02915", 3): "car", ("02915", 7): "car", ("02995", 5): "car", ("03075", 2): "car",
    ("03075", 5): None, ("03155", 9): None, ("03155", 10): "car",
}


def main():
    (HERE / "hem" / "corrections.json").write_text(json.dumps(HEM, indent=1), encoding="utf-8")
    out = {}
    for (f, i), c in J.items():
        e = out.setdefault(f"junction_{f}.jpg", {"drop": [], "cls": {}})
        if c is None:
            e["drop"].append(i)
        else:
            e["cls"][str(i)] = c
    (HERE / "junction" / "corrections.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(len(HEM), "hem frames;", len(J), "junction boxes changed")


if __name__ == "__main__":
    main()
