import random
from dataclasses import dataclass
from collections import defaultdict
from io import BytesIO

import pandas as pd
import streamlit as st


# =============================
# 基礎資料結構
# =============================

@dataclass
class Seat:
    id: int
    row: int  # 列（前後）
    col: int  # 行（左右）


@dataclass
class StudentConstraint:
    id: int
    allowed_rows: set | None
    allowed_cols: set | None
    must_be_adjacent_to: set
    must_not_adjacent_to: set


# =============================
# 座位佈局與鄰近關係
# =============================

def build_default_seats() -> list[Seat]:
    """
    預設座位佈局：
    - 基本為 6 列 x 6 行，共 36 個座位
    - 第二行多一個座位（示例放在第 7 列第 2 行）→ 總共 37 個座位

    若你的實際教室座位位置不同，可以直接修改這個函式。
    """
    seats: list[Seat] = []
    seat_id = 1

    # 6x6 = 36 座位
    for r in range(1, 7):      # 列
        for c in range(1, 7):  # 行
            seats.append(Seat(id=seat_id, row=r, col=c))
            seat_id += 1

    # 第二行多出一個座位（可依實際情況調整 row）
    seats.append(Seat(id=seat_id, row=7, col=2))

    return seats


def build_adjacency_maps(seats: list[Seat]):
    """
    建立：
    - adjacent_lr: 只算左右相鄰（同一列，col 差 1）→ 用於「相鄰」條件
    - adjacent_9: 九宮格（dr <= 1, dc <= 1）→ 用於「不相鄰」嚴格版
    """
    adjacent_lr = defaultdict(set)
    adjacent_9 = defaultdict(set)

    for s1 in seats:
        for s2 in seats:
            if s1.id == s2.id:
                continue
            dr = abs(s1.row - s2.row)
            dc = abs(s1.col - s2.col)

            # 左右相鄰：同一列，行差 1
            if dr == 0 and dc == 1:
                adjacent_lr[s1.id].add(s2.id)

            # 九宮格鄰近
            if dr <= 1 and dc <= 1:
                adjacent_9[s1.id].add(s2.id)

    return adjacent_lr, adjacent_9


# =============================
# 約束檢查與解題器
# =============================

def is_seat_allowed_for_student(seat: Seat, sc: StudentConstraint | None) -> bool:
    if sc is None:
        return True
    if sc.allowed_rows is not None and seat.row not in sc.allowed_rows:
        return False
    if sc.allowed_cols is not None and seat.col not in sc.allowed_cols:
        return False
    return True


def check_partial_constraints(
    assignments: dict[int, int],  # student_id -> seat_id
    student_id: int,
    seat_id: int,
    constraints: dict[int, StudentConstraint],
    adjacent_lr: dict[int, set[int]],
    adjacent_9: dict[int, set[int]],
    use_strict_non_adjacent: bool,
) -> bool:
    """
    檢查目前階段把 student_id 排到 seat_id 是否會違反任何限制。
    - 相鄰：只用 adjacent_lr（左右）
    - 不相鄰（嚴格版）：用 adjacent_9（九宮格）
    - 不相鄰（寬鬆版）：用 adjacent_lr（左右不鄰）
    """
    sc = constraints.get(student_id)

    # 1. row / col 限制
    seat = st.session_state["seat_by_id"][seat_id]
    if not is_seat_allowed_for_student(seat, sc):
        return False

    # 2. 本人對別人的「必須相鄰 / 不相鄰」
    if sc is not None:
        # 必須相鄰（左右）
        for other in sc.must_be_adjacent_to:
            if other in assignments:
                other_seat = assignments[other]
                if other_seat not in adjacent_lr[seat_id]:
                    return False

        # 不相鄰
        for other in sc.must_not_adjacent_to:
            if other in assignments:
                other_seat = assignments[other]
                if use_strict_non_adjacent:
                    # 嚴格版：九宮格不能有
                    if other_seat in adjacent_9[seat_id]:
                        return False
                else:
                    # 寬鬆版：左右不能相鄰（只看左右）
                    if other_seat in adjacent_lr[seat_id]:
                        return False

    # 3. 反向檢查：別人對我也可能有 must_be / must_not
    for other, other_seat in assignments.items():
        osc = constraints.get(other)
        if osc is None:
            continue

        # 對方要求跟我相鄰（左右）
        if student_id in osc.must_be_adjacent_to:
            if other_seat not in adjacent_lr[seat_id]:
                return False

        # 對方要求與我不相鄰
        if student_id in osc.must_not_adjacent_to:
            if use_strict_non_adjacent:
                if seat_id in adjacent_9[other_seat]:
                    return False
            else:
                if seat_id in adjacent_lr[other_seat]:
                    return False

    return True


def solve_one_assignment(
    students: list[int],
    seats: list[Seat],
    constraints: dict[int, StudentConstraint],
    adjacent_lr: dict[int, set[int]],
    adjacent_9: dict[int, set[int]],
    use_strict_non_adjacent: bool,
) -> dict[int, int] | None:
    """
    回傳 assignments: student_id -> seat_id
    找不到解則回傳 None。
    """
    # 排序學生：限制多者優先
    def constraint_score(sid: int) -> int:
        sc = constraints.get(sid)
        if sc is None:
            return 0
        score = 0
        if sc.allowed_rows:
            score += 2
        if sc.allowed_cols:
            score += 2
        score += 3 * len(sc.must_be_adjacent_to)
        score += 3 * len(sc.must_not_adjacent_to)
        return score

    students_sorted = sorted(students, key=constraint_score, reverse=True)

    assignments: dict[int, int] = {}

    def backtrack(idx: int) -> bool:
        if idx == len(students_sorted):
            return True

        sid = students_sorted[idx]
        sc = constraints.get(sid)

        # 找出這個學生可用的候選座位
        candidate_seats: list[int] = []
        for seat in seats:
            if seat.id in assignments.values():
                continue
            if not is_seat_allowed_for_student(seat, sc):
                continue
            candidate_seats.append(seat.id)

        # 打亂候選順序，讓結果更隨機
        random.shuffle(candidate_seats)

        for seat_id in candidate_seats:
            if not check_partial_constraints(
                assignments,
                sid,
                seat_id,
                constraints,
                adjacent_lr,
                adjacent_9,
                use_strict_non_adjacent,
            ):
                continue
            assignments[sid] = seat_id
            if backtrack(idx + 1):
                return True
            del assignments[sid]

        return False

    ok = backtrack(0)
    return assignments if ok else None


def generate_multiple_layouts(
    num_layouts: int,
    students: list[int],
    seats: list[Seat],
    constraints: dict[int, StudentConstraint],
    adjacent_lr: dict[int, set[int]],
    adjacent_9: dict[int, set[int]],
    use_strict_non_adjacent: bool,
) -> list[dict[int, int]]:
    """
    一次產生多個不同的座位表。
    """
    layouts: list[dict[int, int]] = []
    seen = set()
    attempts = 0
    max_attempts = num_layouts * 30  # 安全上限

    while len(layouts) < num_layouts and attempts < max_attempts:
        attempts += 1
        random.shuffle(students)
        result = solve_one_assignment(
            students,
            seats,
            constraints,
            adjacent_lr,
            adjacent_9,
            use_strict_non_adjacent,
        )
        if result is None:
            continue
        key = tuple(sorted(result.items()))
        if key in seen:
            continue
        seen.add(key)
        layouts.append(result)

    return layouts


# =============================
# Excel 匯出
# =============================

def create_excel_file(
    layouts: list[dict[int, int]],
    seats: list[Seat],
    students_info: dict[int, str],
) -> BytesIO:
    """
    每張座位表一個 sheet，以 row/col 排出座位佈局。
    儲存內容格式：「座號 姓名」。
    """
    output = BytesIO()
    seat_by_id = {s.id: s for s in seats}

    # 取得佈局大小（max row / col）
    max_row = max(s.row for s in seats)
    max_col = max(s.col for s in seats)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for i, layout in enumerate(layouts, start=1):
            # 建立空白 DataFrame
            df = pd.DataFrame("", index=range(1, max_row + 1), columns=range(1, max_col + 1))

            for sid, seat_id in layout.items():
                seat = seat_by_id[seat_id]
                name = students_info.get(sid, "")
                df.at[seat.row, seat.col] = f"{sid} {name}"

            sheet_name = f"座位表_{i}"
            df.to_excel(writer, sheet_name=sheet_name)

    output.seek(0)
    return output


# =============================
# Streamlit UI 狀態
# =============================

def init_session_state(seats: list[Seat]):
    if "students_df" not in st.session_state:
        # 預設 37 人，座號 1~37，姓名欄位留空讓使用者填
        st.session_state["students_df"] = pd.DataFrame(
            {
                "座號": list(range(1, len(seats) + 1)),
                "姓名": ["" for _ in range(len(seats))],
            }
        )
    if "constraints_df" not in st.session_state:
        st.session_state["constraints_df"] = pd.DataFrame(
            {
                "座號": list(range(1, len(seats) + 1)),
                "允許列（用逗號分隔，如 1,2,3）": ["" for _ in range(len(seats))],
                "允許行（用逗號分隔，如 1,3,5）": ["" for _ in range(len(seats))],
            }
        )
    if "adjacency_rules" not in st.session_state:
        # 每筆：{"座號A": int, "座號B": int, "類型": "相鄰" 或 "不相鄰"}
        st.session_state["adjacency_rules"] = []
    if "layouts" not in st.session_state:
        st.session_state["layouts"] = []
    if "seat_by_id" not in st.session_state:
        st.session_state["seat_by_id"] = {s.id: s for s in seats}


def parse_rowcol_set(value: str) -> set | None:
    value = (value or "").strip()
    if not value:
        return None
    parts = [v.strip() for v in value.split(",") if v.strip()]
    try:
        nums = {int(v) for v in parts}
        return nums if nums else None
    except ValueError:
        return None


def build_constraints(
    students: list[int],
    constraints_df: pd.DataFrame,
    adjacency_rules: list[dict],
) -> dict[int, StudentConstraint]:
    # 先建立基本 row/col 限制
    base: dict[int, StudentConstraint] = {}
    row_map = {int(r["座號"]): r["允許列（用逗號分隔，如 1,2,3）"] for _, r in constraints_df.iterrows()}
    col_map = {int(r["座號"]): r["允許行（用逗號分隔，如 1,3,5）"] for _, r in constraints_df.iterrows()}

    for sid in students:
        allowed_rows = parse_rowcol_set(row_map.get(sid, ""))
        allowed_cols = parse_rowcol_set(col_map.get(sid, ""))
        base[sid] = StudentConstraint(
            id=sid,
            allowed_rows=allowed_rows,
            allowed_cols=allowed_cols,
            must_be_adjacent_to=set(),
            must_not_adjacent_to=set(),
        )

    # 根據 adjacency_rules 填入 must_be / must_not
    for rule in adjacency_rules:
        a = int(rule["座號A"])
        b = int(rule["座號B"])
        t = rule["類型"]
        if a not in base or b not in base:
            continue
        if t == "相鄰":
            base[a].must_be_adjacent_to.add(b)
            base[b].must_be_adjacent_to.add(a)
        elif t == "不相鄰":
            base[a].must_not_adjacent_to.add(b)
            base[b].must_not_adjacent_to.add(a)

    return base


# =============================
# 主程式（UI）
# =============================

def main():
    st.set_page_config(page_title="亂數排座位表產生器", layout="wide")
    st.title("🎲 亂數排座位表產生器（6x6 + 第二行 7 座）")

    seats = build_default_seats()
    adjacent_lr, adjacent_9 = build_adjacency_maps(seats)
    init_session_state(seats)

    max_row = max(s.row for s in seats)
    max_col = max(s.col for s in seats)

    st.markdown(
        """
        功能說明：
        - 座位佈局：**6x6 格局 + 第二行多一個座位，共 37 座**
        - 支援每位學生：
          - 限制只能坐在哪些「列」（前後）
          - 限制只能坐在哪些「行」（左右）
        - 支援指定兩位學生「相鄰（只算左右）」或「不相鄰（九宮格 / 左右）」
        - 一次產生多張不同座位表，並可匯出 Excel。
        """
    )

    # 側邊欄：基本設定
    with st.sidebar:
        st.header("⚙️ 基本設定")
        num_layouts = st.number_input("要產生幾張座位表？", min_value=1, max_value=20, value=5, step=1)
        st.markdown("---")
        st.markdown(
            """
            **不相鄰判定：**
            1. 先嘗試「九宮格內沒有對方」  
            2. 若太嚴格找不到解，會自動改成「左右不相鄰」較寬鬆版本
            """
        )

    # 1. 學生名單
    st.subheader("1️⃣ 學生名單")
    st.markdown("請輸入或貼上學生名單（座號需為整數，預設為 1~37）。")
    students_df = st.data_editor(
        st.session_state["students_df"],
        num_rows="fixed",
        use_container_width=True,
        key="students_editor",
    )
    st.session_state["students_df"] = students_df

    try:
        student_ids = [int(v) for v in students_df["座號"].tolist()]
    except ValueError:
        st.error("❌ 座號欄位必須全部為整數，請確認。")
        return

    if len(set(student_ids)) != len(student_ids):
        st.error("❌ 座號欄位有重複，請修正。")
        return

    # 2. 個別座位限制（列 / 行） + 快捷設定
    st.subheader("2️⃣ 個別座位限制（列 / 行）")
    st.markdown(
        """
        - 「允許列」與「允許行」可以填入多個數字，以逗號分隔，例如：`1,2,3`  
        - 留白代表**不限制**。  
        - 下方提供「快捷設定」，可以一次把一批學生限定在前 n 列、後 n 列、最左 n 行、最右 n 行。
        """
    )
    constraints_df = st.data_editor(
        st.session_state["constraints_df"],
        num_rows="fixed",
        use_container_width=True,
        key="constraints_editor",
    )
    st.session_state["constraints_df"] = constraints_df

    # ⚡ 快捷設定：前 n 列 / 後 n 列 / 最左 n 行 / 最右 n 行
    with st.expander("⚡ 批次套用列 / 行快捷限制"):
        st.markdown("選擇一批學生，一鍵套用指定列／行範圍。")
        selected_students = st.multiselect(
            "選擇要套用的座號（可複選）",
            options=student_ids,
            default=[],
        )

        col_row, col_col = st.columns(2)

        # 列快捷
        with col_row:
            st.markdown("**列（前後）快捷設定**")
            row_mode = st.selectbox(
                "列快捷類型",
                ["不套用", "前 n 列", "後 n 列"],
                key="row_mode",
            )
            row_n = st.number_input(
                "n（列數）",
                min_value=1,
                max_value=max_row,
                value=1,
                step=1,
                key="row_n",
            )
            if st.button("套用到選擇學生（列）"):
                if not selected_students:
                    st.warning("請先選擇至少一位學生。")
                else:
                    if row_mode == "不套用":
                        new_val = ""
                    elif row_mode == "前 n 列":
                        rows = list(range(1, min(row_n, max_row) + 1))
                        new_val = ",".join(str(r) for r in rows)
                    else:  # 後 n 列
                        start = max_row - row_n + 1
                        start = max(start, 1)
                        rows = list(range(start, max_row + 1))
                        new_val = ",".join(str(r) for r in rows)

                    for sid in selected_students:
                        idx = constraints_df["座號"] == sid
                        constraints_df.loc[idx, "允許列（用逗號分隔，如 1,2,3）"] = new_val

                    st.session_state["constraints_df"] = constraints_df
                    st.success("已套用列快捷設定。")
                    st.rerun()

        # 行快捷
        with col_col:
            st.markdown("**行（左右）快捷設定**")
            col_mode = st.selectbox(
                "行快捷類型",
                ["不套用", "最左邊 n 行", "最右邊 n 行"],
                key="col_mode",
            )
            col_n = st.number_input(
                "n（行數）",
                min_value=1,
                max_value=max_col,
                value=1,
                step=1,
                key="col_n",
            )
            if st.button("套用到選擇學生（行）"):
                if not selected_students:
                    st.warning("請先選擇至少一位學生。")
                else:
                    if col_mode == "不套用":
                        new_val = ""
                    elif col_mode == "最左邊 n 行":
                        cols = list(range(1, min(col_n, max_col) + 1))
                        new_val = ",".join(str(c) for c in cols)
                    else:  # 最右邊 n 行
                        start = max_col - col_n + 1
                        start = max(start, 1)
                        cols = list(range(start, max_col + 1))
                        new_val = ",".join(str(c) for c in cols)

                    for sid in selected_students:
                        idx = constraints_df["座號"] == sid
                        constraints_df.loc[idx, "允許行（用逗號分隔，如 1,3,5）"] = new_val

                    st.session_state["constraints_df"] = constraints_df
                    st.success("已套用行快捷設定。")
                    st.experimental_rerun()

    # 3. 相鄰 / 不相鄰 批次設定
    st.subheader("3️⃣ 相鄰 / 不相鄰 條件設定（可一次輸入多筆）")
    st.markdown(
        """
        - **相鄰**：只算左右相鄰（同一列、行相差 1），前後不算。  
        - **不相鄰**：系統會先以「九宮格」判定，若太嚴格則退而求其次只看左右。  
        - 下表可以直接新增多列，或從 Excel 貼上多筆 `座號A / 座號B / 類型`。
        """
    )

    # 讓使用者直接編輯一整張規則表
    if st.session_state["adjacency_rules"]:
        rules_df = pd.DataFrame(st.session_state["adjacency_rules"])
    else:
        rules_df = pd.DataFrame(columns=["座號A", "座號B", "類型"])

    rules_df = st.data_editor(
        rules_df,
        num_rows="dynamic",
        use_container_width=True,
        key="rules_editor",
        column_config={
            "類型": st.column_config.SelectboxColumn(
                "類型",
                options=["相鄰", "不相鄰"],
                required=True,
            )
        },
    )

    # 清洗並存回 session_state
    new_rules: list[dict] = []
    for _, row in rules_df.iterrows():
        a = row.get("座號A")
        b = row.get("座號B")
        t = row.get("類型")
        try:
            a = int(a)
            b = int(b)
        except (TypeError, ValueError):
            continue
        if t not in ["相鄰", "不相鄰"]:
            continue
        if a == b:
            continue
        if a not in student_ids or b not in student_ids:
            # 若不在目前名單中就略過
            continue
        new_rules.append({"座號A": a, "座號B": b, "類型": t})

    st.session_state["adjacency_rules"] = new_rules

    if not new_rules:
        st.info("目前尚未設定任何相鄰 / 不相鄰條件。")

    # 4. 生成座位表
    st.subheader("4️⃣ 產生亂數座位表")
    generate_clicked = st.button("🚀 生成座位表")

    if generate_clicked:
        # 準備學生資訊
        students_info = {
            int(row["座號"]): str(row["姓名"])
            for _, row in students_df.iterrows()
        }

        # 準備約束條件
        constraints = build_constraints(
            student_ids,
            st.session_state["constraints_df"],
            st.session_state["adjacency_rules"],
        )

        with st.spinner("嘗試產生符合所有條件的座位表（九宮格不相鄰）..."):
            strict_layouts = generate_multiple_layouts(
                num_layouts,
                students=student_ids.copy(),
                seats=seats,
                constraints=constraints,
                adjacent_lr=adjacent_lr,
                adjacent_9=adjacent_9,
                use_strict_non_adjacent=True,
            )

        layouts = strict_layouts
        used_relaxed = False

        if not layouts:
            st.warning("使用九宮格不相鄰條件找不到解，改用較寬鬆的『左右不相鄰』再試一次。")
            with st.spinner("改用左右不相鄰重新嘗試生成..."):
                relaxed_layouts = generate_multiple_layouts(
                    num_layouts,
                    students=student_ids.copy(),
                    seats=seats,
                    constraints=constraints,
                    adjacent_lr=adjacent_lr,
                    adjacent_9=adjacent_9,
                    use_strict_non_adjacent=False,
                )
            layouts = relaxed_layouts
            used_relaxed = True

        if not layouts:
            st.error("❌ 在目前設定下無法產生任何有效座位表，請放寬部分條件後再試。")
        else:
            st.session_state["layouts"] = layouts
            if used_relaxed:
                st.info(f"✅ 已成功產生 {len(layouts)} 張座位表（使用 **左右不相鄰** 的較寬鬆條件）。")
            else:
                st.success(f"✅ 已成功產生 {len(layouts)} 張座位表（使用 **九宮格不相鄰** 的嚴格條件）。")

    # 5. 顯示與下載結果
    layouts = st.session_state.get("layouts", [])
    if layouts:
        st.subheader("5️⃣ 座位表預覽")

        seat_by_id = st.session_state["seat_by_id"]

        tabs = st.tabs([f"座位表 {i+1}" for i in range(len(layouts))])

        for tab, layout in zip(tabs, layouts):
            with tab:
                st.markdown("#### 座位圖（每格：座號 姓名）")

                # 建立「座位 → 學生」反查表
                seat_to_student: dict[int, int] = {seat_id: sid for sid, seat_id in layout.items()}

                for r in range(1, max_row + 1):
                    cols = st.columns(max_col)
                    for c in range(1, max_col + 1):
                        # 找出這個位置是否有座位
                        seat_id = None
                        for s in seats:
                            if s.row == r and s.col == c:
                                seat_id = s.id
                                break
                        if seat_id is None:
                            with cols[c - 1]:
                                st.markdown("⬜️")
                        else:
                            with cols[c - 1]:
                                sid = seat_to_student.get(seat_id)
                                if sid is None:
                                    st.markdown("⬜️（無人）")
                                else:
                                    name = students_info.get(sid, "")
                                    st.markdown(f"🪑 **{sid}**<br/>{name}", unsafe_allow_html=True)

        # Excel 下載
        st.subheader("📥 匯出 Excel")
        students_info = {
            int(row["座號"]): str(row["姓名"])
            for _, row in st.session_state["students_df"].iterrows()
        }
        excel_bytes = create_excel_file(layouts, seats, students_info)
        st.download_button(
            label="下載座位表 Excel",
            data=excel_bytes,
            file_name="座位表_亂數排班.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )



if __name__ == "__main__":
    main()
