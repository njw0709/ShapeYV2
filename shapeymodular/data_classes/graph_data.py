from dataclasses import dataclass
import numpy as np
from typing import Union, List, Dict
import typing
from functools import reduce
import shapeymodular.utils as utils


@dataclass
class GraphData:
    x: Union[np.ndarray, str]
    y: Union[np.ndarray, str]
    x_label: str
    y_label: str
    data: Union[np.ndarray, str]
    label: str
    supplementary_data: Union[Dict[str, np.ndarray], None] = None


@dataclass
class GraphDataGroup:
    data: List[GraphData]

    def __post_init__(self):
        x_label = self.data[0].x_label
        y_label = self.data[0].y_label
        list_x = []
        list_y = []
        for graph_data in self.data:
            if x_label == graph_data.x_label:
                self.x_label = x_label
                list_x.append(graph_data.x)
            else:
                self.x_label = None
                list_x = []
            if y_label == graph_data.y_label:
                self.y_label = y_label
                list_y.append(graph_data.y)
            else:
                self.y_label = None
                list_y = []
        if self.x_label is not None:
            self.x = reduce(np.union1d, list_x)
        else:
            self.x = None
        if self.y_label is not None:
            self.y = reduce(np.union1d, list_y)
        else:
            self.y = None

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def compute_statistics(self) -> None:
        assert self.x is not None
        assert self.y is not None
        assert self.x_label is not None
        assert self.y_label is not None
        labels = [graph_data.label for graph_data in self.data]
        try:
            assert (
                "mean" not in labels
                and "std" not in labels
                and "quantile_75" not in labels
                and "quantile_25" not in labels
            )
        except AssertionError:
            print("Statistics already computed")
            return
        list_all_data = [
            typing.cast(np.ndarray, graph_data.data) for graph_data in self.data
        ]
        mean = np.mean(
            list_all_data,
            axis=0,
        )
        std = np.std(list_all_data, axis=0, ddof=1)
        quantile_75 = np.quantile(list_all_data, 0.75, axis=0)
        quantile_25 = np.quantile(list_all_data, 0.25, axis=0)
        list_stats = [mean, std, quantile_75, quantile_25]
        labels = ["mean", "std", "quantile_75", "quantile_25"]
        statistics = [
            GraphData(
                x=self.x,
                y=self.y,
                x_label=self.x_label,
                y_label=self.y_label,
                data=data,
                label=labels[i],
            )
            for i, data in enumerate(list_stats)
        ]
        self.extend(GraphDataGroup(statistics))

    def extend(self, graph_data_group: "GraphDataGroup"):
        self.data.extend(graph_data_group.data)
        self.__post_init__()
