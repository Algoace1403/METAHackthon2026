"""Seed-based deterministic data corruption pipeline for data cleaning OpenEnv."""

from __future__ import annotations

import copy
import random
import uuid
from typing import Any, Optional


# Name variant dictionary for duplicate noise
NAME_VARIANTS: dict[str, list[str]] = {
    "Robert": ["Rob", "Bob"],
    "William": ["Will", "Bill", "Wm"],
    "Elizabeth": ["Liz", "Beth"],
    "Jennifer": ["Jen", "Jenny"],
    "Michael": ["Mike"],
    "James": ["Jim"],
    "Katherine": ["Kate", "Kathy"],
    "Richard": ["Rick", "Rich"],
}

# Visually similar character substitutions
VISUAL_SIMILAR: dict[str, str] = {
    "o": "0",
    "O": "0",
    "l": "1",
    "I": "1",
    "s": "5",
    "S": "5",
    "z": "2",
    "Z": "2",
    "g": "9",
    "B": "8",
}

# Date format pool for format randomization
DATE_FORMATS: list[str] = [
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%Y-%m-%d",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%m-%d-%Y",
    "%Y/%m/%d",
    "%d.%m.%Y",
    "%B %d %Y",
]

# State abbreviation to full name mapping
STATE_ABBREVIATIONS: dict[str, str] = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}

# Value variation mappings
VALUE_VARIATIONS: dict[str, list[str]] = {
    "Engineering": ["Eng", "Eng.", "ENGINEERING", "engineering"],
    "Marketing": ["Mktg", "Mktg.", "MARKETING", "marketing"],
    "Human Resources": ["HR", "H.R.", "HumanResources"],
    "Information Technology": ["IT", "I.T.", "InfoTech"],
    "Finance": ["Fin", "Fin.", "FINANCE", "finance"],
    "Sales": ["Sls", "SALES", "sales"],
    "Operations": ["Ops", "OPERATIONS", "operations"],
    "Research": ["R&D", "Res", "RESEARCH"],
    "Accounting": ["Acct", "Acct.", "ACCOUNTING"],
    "Management": ["Mgmt", "Mgmt.", "MANAGEMENT"],
}

# Address abbreviation mappings
ADDRESS_ABBREVIATIONS: dict[str, list[str]] = {
    "Street": ["St.", "St", "Str."],
    "Avenue": ["Ave.", "Ave", "Av."],
    "Boulevard": ["Blvd.", "Blvd"],
    "Drive": ["Dr.", "Dr"],
    "Lane": ["Ln.", "Ln"],
    "Road": ["Rd.", "Rd"],
    "Court": ["Ct.", "Ct"],
    "Place": ["Pl.", "Pl"],
    "Apartment": ["Apt.", "Apt", "Apt #"],
    "Suite": ["Ste.", "Ste", "Suite #"],
    "Building": ["Bldg.", "Bldg"],
    "Floor": ["Fl.", "Fl"],
    "North": ["N.", "N"],
    "South": ["S.", "S"],
    "East": ["E.", "E"],
    "West": ["W.", "W"],
}


def _make_row_id(rng: random.Random) -> str:
    """Generate a deterministic UUID-like row id."""
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


class DataCorruptor:
    """Deterministic data corruption pipeline.

    Uses a seeded random.Random instance so the same seed always
    produces the same corrupted output.

    Each handler supports TWO spec formats:
      1. **Explicit targeting** (primary): specific row indices, fields,
         and values are named in the spec.
      2. **Generic/probabilistic** (fallback): columns + probability,
         matching the legacy format.

    When explicit keys (``targets``, ``row_indices``, ``source_indices``,
    ``pairs``, ``mapping``) are present, the handler uses them.  Otherwise
    it falls back to the generic probabilistic path.
    """

    def __init__(self, seed: int) -> None:
        self._rng: random.Random = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def corrupt(
        self,
        clean_data: list[dict[str, Any]],
        corruptions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply a list of corruption specs to *clean_data*.

        Each corruption spec is a dict with at least a ``"type"`` key that
        maps to one of the ``_corrupt_*`` handler methods.  Additional keys
        are forwarded as keyword arguments to the handler.

        Returns a new list of rows (the original is never mutated).
        """
        data = copy.deepcopy(clean_data)

        for spec in corruptions:
            handler_name = f"_corrupt_{spec['type']}"
            handler = getattr(self, handler_name, None)
            if handler is None:
                raise ValueError(f"Unknown corruption type: {spec['type']}")
            data = handler(data, spec)

        return data

    # ------------------------------------------------------------------
    # No-op handlers
    # ------------------------------------------------------------------

    def _corrupt_valid_unusual(
        self,
        data: list[dict[str, Any]],
        spec: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """No-op: valid_unusual entries are documentary annotations, not corruptions."""
        return data

    # ------------------------------------------------------------------
    # Corruption handlers
    # ------------------------------------------------------------------

    def _corrupt_char_swap(
        self,
        data: list[dict[str, Any]],
        spec: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Swap adjacent chars or replace with visually similar characters.

        Explicit spec:
            targets: list[dict] — each dict has "row_idx" and "field"
            mode: str — "swap" | "visual" | "both" (default "both")

        Generic (fallback) spec:
            columns: list[str] — columns to target
            probability: float — per-cell probability (default 0.3)
            mode: str — "swap" | "visual" | "both" (default "both")
        """
        mode: str = spec.get("mode", "both")
        targets: list[dict[str, Any]] | None = spec.get("targets")

        if targets is not None:
            for target in targets:
                row_idx: int = target["row_idx"]
                field: str = target["field"]
                if row_idx >= len(data):
                    continue
                row = data[row_idx]
                if field not in row or row[field] is None:
                    continue

                value = str(row[field])
                if len(value) < 2:
                    continue

                row[field] = self._apply_char_swap(value, mode)
        else:
            columns: list[str] = spec.get("columns", [])
            probability: float = spec.get("probability", 0.3)

            for row in data:
                for col in columns:
                    if col not in row or row[col] is None:
                        continue
                    if self._rng.random() > probability:
                        continue

                    value = str(row[col])
                    if len(value) < 2:
                        continue

                    row[col] = self._apply_char_swap(value, mode)

        return data

    def _corrupt_format_randomize(
        self,
        data: list[dict[str, Any]],
        spec: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Convert date strings to random formats.

        Explicit spec:
            column: str — the date column to target
            row_indices: list[int] — specific rows to corrupt
            source_format: str (default "%Y-%m-%d")

        Generic (fallback) spec:
            columns: list[str] — date columns to target
            probability: float — per-cell probability (default 0.5)
            source_format: str (default "%Y-%m-%d")
        """
        from datetime import datetime

        source_format: str = spec.get("source_format", "%Y-%m-%d")
        row_indices: list[int] | None = spec.get("row_indices")

        if row_indices is not None:
            column: str = spec.get("column", "")
            for idx in row_indices:
                if idx >= len(data):
                    continue
                row = data[idx]
                if column not in row or row[column] is None:
                    continue
                try:
                    dt = datetime.strptime(str(row[column]), source_format)
                    new_fmt = self._rng.choice(DATE_FORMATS)
                    row[column] = dt.strftime(new_fmt)
                except ValueError:
                    pass
        else:
            columns: list[str] = spec.get("columns", [])
            probability: float = spec.get("probability", 0.5)

            for row in data:
                for col in columns:
                    if col not in row or row[col] is None:
                        continue
                    if self._rng.random() > probability:
                        continue
                    try:
                        dt = datetime.strptime(str(row[col]), source_format)
                        new_fmt = self._rng.choice(DATE_FORMATS)
                        row[col] = dt.strftime(new_fmt)
                    except ValueError:
                        pass

        return data

    def _corrupt_null_inject(
        self,
        data: list[dict[str, Any]],
        spec: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Set specific cells to None.

        Explicit spec:
            targets: list[dict] — each dict has "row_idx" and "field"

        Generic (fallback) spec:
            columns: list[str] — columns to target
            probability: float — per-cell probability (default 0.1)
        """
        targets: list[dict[str, Any]] | None = spec.get("targets")

        if targets is not None:
            for target in targets:
                row_idx: int = target["row_idx"]
                field: str = target["field"]
                if row_idx >= len(data):
                    continue
                if field in data[row_idx]:
                    data[row_idx][field] = None
        else:
            columns: list[str] = spec.get("columns", [])
            probability: float = spec.get("probability", 0.1)

            for row in data:
                for col in columns:
                    if col not in row:
                        continue
                    if self._rng.random() < probability:
                        row[col] = None

        return data

    def _corrupt_case_corrupt(
        self,
        data: list[dict[str, Any]],
        spec: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Randomize string values to lower/upper/mixed case.

        Explicit spec:
            targets: list[dict] — each dict has "row_idx" and "field"

        Generic (fallback) spec:
            columns: list[str]
            probability: float (default 0.3)
        """
        targets: list[dict[str, Any]] | None = spec.get("targets")

        if targets is not None:
            for target in targets:
                row_idx: int = target["row_idx"]
                field: str = target["field"]
                if row_idx >= len(data):
                    continue
                row = data[row_idx]
                if field not in row or row[field] is None:
                    continue

                row[field] = self._apply_case_corrupt(str(row[field]))
        else:
            columns: list[str] = spec.get("columns", [])
            probability: float = spec.get("probability", 0.3)

            for row in data:
                for col in columns:
                    if col not in row or row[col] is None:
                        continue
                    if self._rng.random() > probability:
                        continue
                    row[col] = self._apply_case_corrupt(str(row[col]))

        return data

    def _corrupt_format_strip(
        self,
        data: list[dict[str, Any]],
        spec: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Strip phone formatting, keeping digits only.

        Explicit spec:
            column: str — the column to strip
            row_indices: list[int] — specific rows to strip

        Generic (fallback) spec:
            columns: list[str]
            probability: float (default 0.4)
        """
        row_indices: list[int] | None = spec.get("row_indices")

        if row_indices is not None:
            column: str = spec.get("column", "")
            for idx in row_indices:
                if idx >= len(data):
                    continue
                row = data[idx]
                if column not in row or row[column] is None:
                    continue
                row[column] = "".join(
                    c for c in str(row[column]) if c.isdigit()
                )
        else:
            columns: list[str] = spec.get("columns", [])
            probability: float = spec.get("probability", 0.4)

            for row in data:
                for col in columns:
                    if col not in row or row[col] is None:
                        continue
                    if self._rng.random() > probability:
                        continue
                    row[col] = "".join(
                        c for c in str(row[col]) if c.isdigit()
                    )

        return data

    def _corrupt_duplicate_with_noise(
        self,
        data: list[dict[str, Any]],
        spec: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Clone rows with name variants and format changes.

        Explicit spec:
            source_indices: list[int] — indices of rows to duplicate
            noise_fields: list[str] — fields on which to apply noise

        Generic (fallback) spec:
            probability: float — per-row probability of duplication (default 0.2)
            name_columns: list[str] — columns containing names to vary
            format_columns: list[str] — columns to reformat (phones, dates)
        """
        source_indices: list[int] | None = spec.get("source_indices")

        if source_indices is not None:
            noise_fields: list[str] = spec.get("noise_fields", [])
            new_rows: list[dict[str, Any]] = []

            for src_idx in source_indices:
                if src_idx >= len(data):
                    continue
                source_row = data[src_idx]
                dup = copy.deepcopy(source_row)
                # Copy _entity_id from source (already in deepcopy)
                # New _row_id will be assigned by the environment in reset()

                for field in noise_fields:
                    if field not in dup or dup[field] is None:
                        continue
                    value = str(dup[field])
                    # Try name variant first, then strip formatting
                    variant = self._apply_name_variant(value)
                    if variant != value:
                        dup[field] = variant
                    else:
                        # Strip formatting as noise
                        digits = "".join(c for c in value if c.isdigit())
                        if digits:
                            dup[field] = digits

                new_rows.append(dup)

            data.extend(new_rows)
        else:
            probability: float = spec.get("probability", 0.2)
            name_columns: list[str] = spec.get("name_columns", [])
            format_columns: list[str] = spec.get("format_columns", [])

            new_rows = []
            for row in data:
                if self._rng.random() > probability:
                    continue

                dup = copy.deepcopy(row)
                dup["_row_id"] = _make_row_id(self._rng)

                for col in name_columns:
                    if col not in dup or dup[col] is None:
                        continue
                    dup[col] = self._apply_name_variant(str(dup[col]))

                for col in format_columns:
                    if col not in dup or dup[col] is None:
                        continue
                    value = str(dup[col])
                    digits = "".join(c for c in value if c.isdigit())
                    if digits:
                        dup[col] = digits

                new_rows.append(dup)

            data.extend(new_rows)

        return data

    def _corrupt_value_variation(
        self,
        data: list[dict[str, Any]],
        spec: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Map values to known variants (e.g. Engineering -> Eng).

        Explicit spec:
            column: str — the column to target
            mapping: dict[str, list[str]] — value -> list of variants

        Generic (fallback) spec:
            columns: list[str]
            probability: float (default 0.3)
            custom_mappings: dict[str, list[str]]
        """
        mapping: dict[str, list[str]] | None = spec.get("mapping")

        if mapping is not None:
            column: str = spec.get("column", "")
            for row in data:
                if column not in row or row[column] is None:
                    continue
                value = str(row[column])
                if value in mapping:
                    row[column] = self._rng.choice(mapping[value])
        else:
            columns: list[str] = spec.get("columns", [])
            probability: float = spec.get("probability", 0.3)
            custom: dict[str, list[str]] = spec.get("custom_mappings", {})
            mappings = {**VALUE_VARIATIONS, **custom}

            for row in data:
                for col in columns:
                    if col not in row or row[col] is None:
                        continue
                    if self._rng.random() > probability:
                        continue
                    value = str(row[col])
                    if value in mappings:
                        row[col] = self._rng.choice(mappings[value])

        return data

    def _corrupt_state_expand(
        self,
        data: list[dict[str, Any]],
        spec: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Expand state abbreviations to full names (CA -> California).

        Explicit spec:
            row_indices: list[int] — specific rows to expand
            column: str — state column (default "state")

        Generic (fallback) spec:
            columns: list[str]
            probability: float (default 0.4)
        """
        row_indices: list[int] | None = spec.get("row_indices")

        if row_indices is not None:
            column: str = spec.get("column", "state")
            for idx in row_indices:
                if idx >= len(data):
                    continue
                row = data[idx]
                if column not in row or row[column] is None:
                    continue
                value = str(row[column]).strip()
                if value.upper() in STATE_ABBREVIATIONS:
                    row[column] = STATE_ABBREVIATIONS[value.upper()]
        else:
            columns: list[str] = spec.get("columns", [])
            probability: float = spec.get("probability", 0.4)

            for row in data:
                for col in columns:
                    if col not in row or row[col] is None:
                        continue
                    if self._rng.random() > probability:
                        continue
                    value = str(row[col]).strip()
                    if value.upper() in STATE_ABBREVIATIONS:
                        row[col] = STATE_ABBREVIATIONS[value.upper()]

        return data

    def _corrupt_duplicate_cluster(
        self,
        data: list[dict[str, Any]],
        spec: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Create clusters of duplicates from source rows.

        Explicit spec:
            source_indices: list[int] — indices of source rows
            cluster_sizes: list[int] — number of duplicates per source
            noise_fields: list[str] — fields on which to apply noise

        Generic (fallback) spec:
            probability: float — per-row probability (default 0.15)
            name_columns: list[str]
            address_columns: list[str]
            phone_columns: list[str]
        """
        source_indices: list[int] | None = spec.get("source_indices")

        if source_indices is not None:
            cluster_sizes: list[int] = spec.get("cluster_sizes", [])
            noise_fields: list[str] = spec.get("noise_fields", [])
            new_rows: list[dict[str, Any]] = []

            for i, src_idx in enumerate(source_indices):
                if src_idx >= len(data):
                    continue
                size = cluster_sizes[i] if i < len(cluster_sizes) else 2
                source_row = data[src_idx]

                for _ in range(size):
                    dup = copy.deepcopy(source_row)
                    # Copy _entity_id from source (already in deepcopy)
                    # New _row_id assigned by environment in reset()

                    for field in noise_fields:
                        if field not in dup or dup[field] is None:
                            continue
                        value = str(dup[field])
                        variant = self._apply_name_variant(value)
                        if variant != value:
                            dup[field] = variant
                        else:
                            # Apply phone reformatting or strip
                            digits = "".join(
                                c for c in value if c.isdigit()
                            )
                            if len(digits) == 10:
                                fmt = self._rng.choice([
                                    f"({digits[:3]}) {digits[3:6]}-{digits[6:]}",
                                    f"{digits[:3]}-{digits[3:6]}-{digits[6:]}",
                                    f"{digits[:3]}.{digits[3:6]}.{digits[6:]}",
                                    digits,
                                    f"+1{digits}",
                                    f"1-{digits[:3]}-{digits[3:6]}-{digits[6:]}",
                                ])
                                dup[field] = fmt
                            elif digits:
                                dup[field] = digits
                            else:
                                dup[field] = self._apply_address_variation(
                                    value
                                )

                    new_rows.append(dup)

            data.extend(new_rows)
        else:
            probability: float = spec.get("probability", 0.15)
            name_columns: list[str] = spec.get("name_columns", [])
            address_columns: list[str] = spec.get("address_columns", [])
            phone_columns: list[str] = spec.get("phone_columns", [])

            new_rows = []
            for row in data:
                if self._rng.random() > probability:
                    continue

                cluster_size = self._rng.randint(2, 3)
                for _ in range(cluster_size):
                    dup = copy.deepcopy(row)
                    dup["_row_id"] = _make_row_id(self._rng)

                    for col in name_columns:
                        if col not in dup or dup[col] is None:
                            continue
                        dup[col] = self._apply_name_variant(str(dup[col]))

                    for col in address_columns:
                        if col not in dup or dup[col] is None:
                            continue
                        dup[col] = self._apply_address_variation(str(dup[col]))

                    for col in phone_columns:
                        if col not in dup or dup[col] is None:
                            continue
                        phone = str(dup[col])
                        digits = "".join(c for c in phone if c.isdigit())
                        if len(digits) == 10:
                            fmt = self._rng.choice([
                                f"({digits[:3]}) {digits[3:6]}-{digits[6:]}",
                                f"{digits[:3]}-{digits[3:6]}-{digits[6:]}",
                                f"{digits[:3]}.{digits[3:6]}.{digits[6:]}",
                                digits,
                                f"+1{digits}",
                                f"1-{digits[:3]}-{digits[3:6]}-{digits[6:]}",
                            ])
                            dup[col] = fmt

                    new_rows.append(dup)

            data.extend(new_rows)

        return data

    def _corrupt_cross_field_corrupt(
        self,
        data: list[dict[str, Any]],
        spec: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Mismatch zip codes with cities by swapping zip values.

        Explicit spec:
            row_indices: list[int] — specific rows whose zips to swap
            zip_column: str (default "zip")

        Generic (fallback) spec:
            zip_column: str (default "zip")
            city_column: str (default "city")
            probability: float (default 0.15)
        """
        zip_column: str = spec.get("zip_column", "zip")
        row_indices: list[int] | None = spec.get("row_indices")

        if row_indices is not None:
            # For explicit targeting: swap zip of each targeted row with
            # a randomly chosen other targeted row.
            valid = [
                idx for idx in row_indices
                if idx < len(data)
                and zip_column in data[idx]
                and data[idx][zip_column] is not None
            ]
            if len(valid) >= 2:
                for idx in valid:
                    others = [i for i in valid if i != idx]
                    swap_idx = self._rng.choice(others)
                    data[idx][zip_column], data[swap_idx][zip_column] = (
                        data[swap_idx][zip_column],
                        data[idx][zip_column],
                    )
        else:
            probability: float = spec.get("probability", 0.15)
            eligible_indices = [
                i for i, row in enumerate(data)
                if zip_column in row and row[zip_column] is not None
            ]

            if len(eligible_indices) < 2:
                return data

            for idx in eligible_indices:
                if self._rng.random() > probability:
                    continue
                swap_idx = self._rng.choice(
                    [i for i in eligible_indices if i != idx]
                )
                data[idx][zip_column], data[swap_idx][zip_column] = (
                    data[swap_idx][zip_column],
                    data[idx][zip_column],
                )

        return data

    def _corrupt_impossible_date(
        self,
        data: list[dict[str, Any]],
        spec: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Create impossible dates: future DOB, visit before birth.

        Explicit spec:
            targets: list[dict] — each dict has:
                "row_idx": int
                "field": str — date field to corrupt
                "corrupt_type": str — "future" | "visit_before_birth"
            source_format: str (default "%Y-%m-%d")

        Generic (fallback) spec:
            dob_column: str (default "dob")
            visit_column: str (optional)
            source_format: str (default "%Y-%m-%d")
            probability: float (default 0.1)
        """
        from datetime import datetime, timedelta

        source_format: str = spec.get("source_format", "%Y-%m-%d")
        targets: list[dict[str, Any]] | None = spec.get("targets")

        if targets is not None:
            for target in targets:
                row_idx: int = target["row_idx"]
                field: str = target["field"]
                corrupt_type: str = target.get("corrupt_type", "future")

                if row_idx >= len(data):
                    continue
                row = data[row_idx]
                if field not in row or row[field] is None:
                    continue

                try:
                    dt = datetime.strptime(str(row[field]), source_format)
                except ValueError:
                    continue

                if corrupt_type == "future":
                    future_offset = self._rng.randint(1, 3650)
                    future_date = datetime.now() + timedelta(
                        days=future_offset
                    )
                    row[field] = future_date.strftime(source_format)
                elif corrupt_type == "visit_before_birth":
                    before_birth = dt - timedelta(
                        days=self._rng.randint(30, 3650)
                    )
                    row[field] = before_birth.strftime(source_format)
        else:
            dob_column: str = spec.get("dob_column", "dob")
            visit_column: Optional[str] = spec.get("visit_column")
            probability: float = spec.get("probability", 0.1)

            for row in data:
                if self._rng.random() > probability:
                    continue
                if dob_column not in row or row[dob_column] is None:
                    continue

                try:
                    dob = datetime.strptime(
                        str(row[dob_column]), source_format
                    )
                except ValueError:
                    continue

                corruption_type = self._rng.choice(
                    ["future_dob", "visit_before_birth"]
                )

                if corruption_type == "future_dob":
                    future_offset = self._rng.randint(1, 3650)
                    future_date = datetime.now() + timedelta(
                        days=future_offset
                    )
                    row[dob_column] = future_date.strftime(source_format)
                elif (
                    corruption_type == "visit_before_birth"
                    and visit_column
                    and visit_column in row
                ):
                    before_birth = dob - timedelta(
                        days=self._rng.randint(30, 3650)
                    )
                    row[visit_column] = before_birth.strftime(source_format)

        return data

    def _corrupt_insurance_id_mismatch(
        self,
        data: list[dict[str, Any]],
        spec: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Assign wrong prefix for insurance provider.

        Explicit spec:
            row_indices: list[int] — specific rows to corrupt
            id_column: str (default "insurance_id")
            provider_column: str (default "insurance_provider")
            prefix_map: dict[str, str] (default standard map)

        Generic (fallback) spec:
            id_column: str (default "insurance_id")
            provider_column: str (default "insurance_provider")
            prefix_map: dict[str, str]
            probability: float (default 0.15)
        """
        id_column: str = spec.get("id_column", "insurance_id")
        provider_column: str = spec.get("provider_column", "insurance_provider")
        prefix_map: dict[str, str] = spec.get("prefix_map", {
            "BlueCross": "BC",
            "Aetna": "AE",
            "UnitedHealth": "UH",
            "Cigna": "CG",
            "Humana": "HM",
            "Kaiser": "KP",
        })
        all_prefixes = list(prefix_map.values())
        if len(all_prefixes) < 2:
            return data

        row_indices: list[int] | None = spec.get("row_indices")

        if row_indices is not None:
            for idx in row_indices:
                if idx >= len(data):
                    continue
                row = data[idx]
                self._swap_insurance_prefix(
                    row, id_column, provider_column, prefix_map, all_prefixes
                )
        else:
            probability: float = spec.get("probability", 0.15)

            for row in data:
                if self._rng.random() > probability:
                    continue
                self._swap_insurance_prefix(
                    row, id_column, provider_column, prefix_map, all_prefixes
                )

        return data

    def _corrupt_null_inject_contextual(
        self,
        data: list[dict[str, Any]],
        spec: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Null out fields that can be inferred from context.

        Explicit spec:
            targets: list[dict] — each dict has "row_idx" and "field"

        Generic (fallback) spec:
            inferable_pairs: list[dict] — each dict has:
                "null_column": str
                "context_column": str
            probability: float (default 0.2)
        """
        targets: list[dict[str, Any]] | None = spec.get("targets")

        if targets is not None:
            for target in targets:
                row_idx: int = target["row_idx"]
                field: str = target["field"]
                if row_idx >= len(data):
                    continue
                if field in data[row_idx]:
                    data[row_idx][field] = None
        else:
            pairs: list[dict[str, str]] = spec.get("inferable_pairs", [])
            probability: float = spec.get("probability", 0.2)

            for row in data:
                for pair in pairs:
                    null_col = pair["null_column"]
                    context_col = pair["context_column"]
                    if null_col not in row or context_col not in row:
                        continue
                    if row[context_col] is None:
                        continue
                    if self._rng.random() > probability:
                        continue
                    row[null_col] = None

        return data

    def _corrupt_false_positive_duplicate(
        self,
        data: list[dict[str, Any]],
        spec: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Mark row pairs as false positives (NOT duplicates).

        No new rows are created.  The rows already exist in the dataset.
        This handler annotates them with a ``_false_positive_pair`` marker
        so the grader knows they should NOT be merged.

        Explicit & generic spec (same format):
            pairs: list[list[int]] — pairs of row indices
            marker_column: str (default "_false_positive_pair")
        """
        pairs: list[list[int]] = spec.get("pairs", [])
        marker_column: str = spec.get("marker_column", "_false_positive_pair")

        pair_id = 0
        for pair in pairs:
            if len(pair) != 2:
                continue
            idx_a, idx_b = pair
            if idx_a < len(data) and idx_b < len(data):
                data[idx_a][marker_column] = f"fp_{pair_id}"
                data[idx_b][marker_column] = f"fp_{pair_id}"
                pair_id += 1

        return data

    def _corrupt_address_variation(
        self,
        data: list[dict[str, Any]],
        spec: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Apply address abbreviation changes (Street -> St., etc.).

        Explicit spec:
            row_indices: list[int] — specific rows to corrupt
            column: str — address column (default "address")

        Generic (fallback) spec:
            columns: list[str]
            probability: float (default 0.4)
        """
        row_indices: list[int] | None = spec.get("row_indices")

        if row_indices is not None:
            column: str = spec.get("column", "address")
            for idx in row_indices:
                if idx >= len(data):
                    continue
                row = data[idx]
                if column not in row or row[column] is None:
                    continue
                row[column] = self._apply_address_variation(str(row[column]))
        else:
            columns: list[str] = spec.get("columns", [])
            probability: float = spec.get("probability", 0.4)

            for row in data:
                for col in columns:
                    if col not in row or row[col] is None:
                        continue
                    if self._rng.random() > probability:
                        continue
                    row[col] = self._apply_address_variation(str(row[col]))

        return data

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_char_swap(self, value: str, mode: str) -> str:
        """Apply a single char-swap or visual-similar corruption."""
        chosen_mode = mode
        if mode == "both":
            chosen_mode = self._rng.choice(["swap", "visual"])

        if chosen_mode == "swap":
            idx = self._rng.randint(0, len(value) - 2)
            chars = list(value)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            return "".join(chars)
        else:  # visual
            replaceable = [
                i for i, c in enumerate(value) if c in VISUAL_SIMILAR
            ]
            if replaceable:
                idx = self._rng.choice(replaceable)
                chars = list(value)
                chars[idx] = VISUAL_SIMILAR[chars[idx]]
                return "".join(chars)
        return value

    def _apply_case_corrupt(self, value: str) -> str:
        """Apply random case corruption to a string value."""
        case_fn = self._rng.choice(["lower", "upper", "mixed"])
        if case_fn == "lower":
            return value.lower()
        elif case_fn == "upper":
            return value.upper()
        else:
            return "".join(
                c.upper() if self._rng.random() > 0.5 else c.lower()
                for c in value
            )

    def _apply_name_variant(self, name: str) -> str:
        """Replace a first name with a known variant if one exists."""
        parts = name.split()
        if not parts:
            return name

        first = parts[0]
        # Check if the first name has known variants
        if first in NAME_VARIANTS:
            variant = self._rng.choice(NAME_VARIANTS[first])
            parts[0] = variant
            return " ".join(parts)

        # Also try case-insensitive match
        for canonical, variants in NAME_VARIANTS.items():
            if first.lower() == canonical.lower():
                variant = self._rng.choice(variants)
                parts[0] = variant
                return " ".join(parts)

        # If no variant found, apply minor perturbation (swap case of first char)
        if len(first) > 1:
            if self._rng.random() < 0.5:
                parts[0] = first[0].lower() + first[1:]
            else:
                parts[0] = first[0].upper() + first[1:]
            return " ".join(parts)

        return name

    def _apply_address_variation(self, address: str) -> str:
        """Replace address terms with abbreviations or vice versa."""
        result = address
        for full_form, abbreviations in ADDRESS_ABBREVIATIONS.items():
            if full_form in result:
                replacement = self._rng.choice(abbreviations)
                result = result.replace(full_form, replacement, 1)
            else:
                # Check if any abbreviation is present and expand it
                for abbr in abbreviations:
                    if abbr in result:
                        if self._rng.random() < 0.5:
                            result = result.replace(abbr, full_form, 1)
                        break

        return result

    def _swap_insurance_prefix(
        self,
        row: dict[str, Any],
        id_column: str,
        provider_column: str,
        prefix_map: dict[str, str],
        all_prefixes: list[str],
    ) -> None:
        """Swap the insurance ID prefix to a wrong one (in-place)."""
        if id_column not in row or provider_column not in row:
            return
        if row[id_column] is None or row[provider_column] is None:
            return

        provider = str(row[provider_column])
        correct_prefix = prefix_map.get(provider)
        if correct_prefix is None:
            return

        wrong_prefixes = [p for p in all_prefixes if p != correct_prefix]
        if not wrong_prefixes:
            return
        wrong_prefix = self._rng.choice(wrong_prefixes)

        current_id = str(row[id_column])
        if current_id.startswith(correct_prefix):
            row[id_column] = wrong_prefix + current_id[len(correct_prefix):]
        else:
            row[id_column] = wrong_prefix + current_id


def generate_dirty_data(
    clean_data: list[dict[str, Any]],
    corruptions: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    """Top-level function: apply corruption pipeline to clean data.

    Args:
        clean_data: List of row dicts representing the clean dataset.
        corruptions: List of corruption spec dicts. Each must contain a
            ``"type"`` key matching one of the DataCorruptor handler names
            (without the ``_corrupt_`` prefix).
        seed: Integer seed for deterministic output.

    Returns:
        A new list of (possibly more) row dicts with corruptions applied.
        The original *clean_data* is never mutated.
    """
    corruptor = DataCorruptor(seed=seed)
    return corruptor.corrupt(clean_data, corruptions)
