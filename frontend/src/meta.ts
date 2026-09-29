/**
 * What the backend's slice endpoints accept (statistics, pooling modes, channel orders) and the text to show for each,
 * from `GET /api/meta`. The lists live in backend/actlens/reducers.py; registering something there is enough for it to
 * show up here. `DEFAULT_META` mirrors the built-ins, so the UI works before the response arrives and against a
 * backend that predates the endpoint.
 */
import { useQuery } from "@tanstack/react-query";
import { getJSON } from "./api";

export interface StatInfo {
  id: string;
  /** Short name (chips, axis titles). */
  label: string;
  /** Name used in menus. */
  long_label: string;
  /** Tooltip. */
  title: string;
  /** Values can be negative (rankings are by |value|). */
  signed: boolean;
  /** The across-layers map uses a diverging colormap centred on zero. */
  diverging: boolean;
}
export interface OptionInfo {
  id: string;
  label: string;
}
export interface Meta {
  /** Reductions along an axis: offered by the across-layers map and the distribution panel. */
  stats: StatInfo[];
  /** Extra across-layers statistics that are not reductions (a single channel). */
  overview_extra: StatInfo[];
  aggs: OptionInfo[];
  orders: OptionInfo[];
}

const stat = (id: string, label: string, title: string, long_label = label, signed = false, diverging = false): StatInfo => ({ id, label, long_label, title, signed, diverging });

export const DEFAULT_META: Meta = {
  stats: [
    stat("norm", "norm", "L2 norm", "L2 norm"),
    stat("absmax", "|max|", "largest absolute value"),
    stat("mean", "mean", "mean (signed)", "mean", true, true),
    stat("std", "std", "standard deviation"),
    stat("kurtosis", "kurtosis", "excess kurtosis (heavy-tailedness)", "excess kurtosis", true),
  ],
  overview_extra: [stat("dim", "single channel", "one channel", "single channel", true, true)],
  aggs: [
    { id: "absmax", label: "abs-max (keeps outliers)" },
    { id: "mean", label: "mean" },
    { id: "max", label: "max" },
    { id: "min", label: "min" },
  ],
  orders: [
    { id: "natural", label: "natural index" },
    { id: "absmax", label: "|max| over tokens ↓" },
    { id: "std", label: "std over tokens ↓" },
    { id: "mean_abs", label: "|mean| over tokens ↓" },
  ],
};

export function useMeta(): Meta {
  const q = useQuery({ queryKey: ["meta"], queryFn: ({ signal }) => getJSON<Meta>("/api/meta", undefined, signal), retry: false });
  return q.data ?? DEFAULT_META;
}

/** Display text of an id from one of the lists; the id itself when the list does not know it. */
export const labelOf = (items: OptionInfo[], id: string): string => items.find((x) => x.id === id)?.label ?? id;
export const statOfId = (stats: StatInfo[], id: string): StatInfo | undefined => stats.find((x) => x.id === id);
