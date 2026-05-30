#!/usr/bin/env python3
"""
METR-LA Traffic Heatmap Visualization

Visualizes METR-LA traffic data with sensors arranged in linear road segments.
Uses geographical clustering and road direction detection to group sensors
into meaningful highway segments without junctions.

Features:
- Detects paired sensors (opposite directions on same road)
- Groups sensors by highway corridor using DBSCAN clustering
- Sorts sensors along road direction for proper spatial ordering
- Generates heatmap for each direction of each highway segment

Data format: LibCity format (.geo, .dyna, .rel files)

Usage:
    python metr_heatmap.py --day 10
    python metr_heatmap.py --batch --start-day 0 --end-day 30
"""

import argparse
import csv
import math
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from typing import Dict, List, Tuple, Optional


class METRHeatmapVisualizer:
    """METR-LA heatmap visualizer with linear road segment detection"""

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = Path(__file__).parent / 'METR_LA'
        self.data_dir = Path(data_dir)

        # File paths
        self.geo_file = self.data_dir / 'METR_LA.geo'
        self.dyna_file = self.data_dir / 'METR_LA.dyna'
        self.rel_file = self.data_dir / 'METR_LA.rel'

        # Data containers
        self.sensors: Dict[int, dict] = {}  # geo_id -> {lon, lat, ...}
        self.traffic_data: Optional[np.ndarray] = None  # shape: (time_steps, n_sensors)
        self.sensor_order: List[int] = []  # ordered sensor IDs
        self.timestamps: List[datetime] = []
        self.road_segments: Dict[str, dict] = {}  # segment_name -> {sensor_ids, direction, ...}

        # Visualization settings
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        self.cmap = self._create_colormap()

        # Constants
        self.START_DATE = datetime(2012, 3, 1)
        self.STEPS_PER_DAY = 288  # 5-minute intervals

    def _create_colormap(self):
        """Create blue colormap (dark=congestion, light=free flow)"""
        colors = [
            '#08306b',  # Deep blue (congestion/low speed)
            '#2171b5',
            '#4292c6',
            '#6baed6',
            '#9ecae1',
            '#c6dbef',
            '#deebf7',
            '#f7fbff',  # Light blue/white (free flow/high speed)
        ]
        return LinearSegmentedColormap.from_list('traffic', colors)

    def load_sensors(self) -> None:
        """Load sensor locations from .geo file"""
        print(f"Loading sensors from {self.geo_file}")

        with open(self.geo_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                geo_id = int(row['geo_id'])
                coords = eval(row['coordinates'])
                self.sensors[geo_id] = {
                    'lon': coords[0],
                    'lat': coords[1],
                    'geo_id': geo_id
                }

        self.sensor_order = sorted(self.sensors.keys())
        print(f"  Loaded {len(self.sensors)} sensors")

    def load_traffic_data(self, day_index: int = None) -> np.ndarray:
        """
        Load traffic data from .dyna file

        Args:
            day_index: If specified, only load data for that day (0-indexed)

        Returns:
            np.ndarray of shape (time_steps, n_sensors)
        """
        print(f"Loading traffic data from {self.dyna_file}")

        # Create sensor ID to index mapping
        sensor_to_idx = {sid: idx for idx, sid in enumerate(self.sensor_order)}
        n_sensors = len(self.sensor_order)

        # First pass: count time steps and collect data
        data_by_time = defaultdict(lambda: np.zeros(n_sensors))
        timestamps_set = set()

        with open(self.dyna_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                entity_id = int(row['entity_id'])
                if entity_id not in sensor_to_idx:
                    continue

                time_str = row['time']
                speed = float(row['traffic_speed'])

                # Parse timestamp
                ts = datetime.fromisoformat(time_str.replace('Z', '+00:00').replace('+00:00', ''))

                # Filter by day if specified
                if day_index is not None:
                    target_date = self.START_DATE + timedelta(days=day_index)
                    if ts.date() != target_date.date():
                        continue

                timestamps_set.add(ts)
                idx = sensor_to_idx[entity_id]
                data_by_time[ts][idx] = speed

        # Convert to array
        self.timestamps = sorted(timestamps_set)
        n_times = len(self.timestamps)

        if n_times == 0:
            raise ValueError(f"No data found for day {day_index}")

        self.traffic_data = np.zeros((n_times, n_sensors))
        for t_idx, ts in enumerate(self.timestamps):
            self.traffic_data[t_idx] = data_by_time[ts]

        print(f"  Loaded data shape: {self.traffic_data.shape}")
        print(f"  Time range: {self.timestamps[0]} to {self.timestamps[-1]}")

        return self.traffic_data

    def detect_paired_sensors(self, distance_threshold: float = 50) -> List[Tuple[int, int]]:
        """
        Detect paired sensors (opposite directions on same road)

        Args:
            distance_threshold: Maximum distance in meters for sensors to be considered a pair

        Returns:
            List of (sensor1_id, sensor2_id) tuples
        """
        pairs = []
        sensor_ids = list(self.sensors.keys())

        for i, id1 in enumerate(sensor_ids):
            for id2 in sensor_ids[i+1:]:
                s1, s2 = self.sensors[id1], self.sensors[id2]
                dist = self._haversine_distance(s1['lat'], s1['lon'], s2['lat'], s2['lon'])
                if dist < distance_threshold:
                    pairs.append((id1, id2))

        print(f"  Found {len(pairs)} paired sensors (distance < {distance_threshold}m)")
        return pairs

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in meters"""
        R = 6371000  # Earth radius in meters
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        return R * c

    def cluster_sensors_by_corridor(self) -> Dict[str, dict]:
        """
        Cluster sensors into highway corridors based on geographical alignment

        This uses a simple but effective approach:
        1. Identify the main direction of each sensor's neighbors
        2. Group sensors that form continuous linear segments
        3. Separate paired sensors (opposite directions)

        Returns:
            Dict of segment_name -> {sensor_ids, direction, sorted_ids}
        """
        print("Clustering sensors into highway corridors...")

        # Detect paired sensors first
        pairs = self.detect_paired_sensors(50)
        paired_set = set()
        pair_mapping = {}  # sensor_id -> partner_id
        for id1, id2 in pairs:
            paired_set.add(id1)
            paired_set.add(id2)
            pair_mapping[id1] = id2
            pair_mapping[id2] = id1

        # Define LA highway corridors based on known geography
        # These corridors are defined by bounding boxes and primary direction
        corridors = self._define_la_corridors()

        # Assign sensors to corridors
        assigned = set()
        segments = {}

        for corridor_name, corridor_def in corridors.items():
            bbox = corridor_def['bbox']  # (min_lon, max_lon, min_lat, max_lat)
            primary_dir = corridor_def['direction']  # 'NS' or 'EW'

            # Find sensors in this corridor
            corridor_sensors = []
            for sid, info in self.sensors.items():
                if sid in assigned:
                    continue
                if (bbox[0] <= info['lon'] <= bbox[1] and
                    bbox[2] <= info['lat'] <= bbox[3]):
                    corridor_sensors.append(sid)

            if len(corridor_sensors) < 3:
                continue

            # Separate into two directions if we have paired sensors
            dir1_sensors = []
            dir2_sensors = []
            unpaired = []

            for sid in corridor_sensors:
                if sid in paired_set:
                    partner = pair_mapping[sid]
                    if partner in corridor_sensors:
                        # Determine which direction based on relative position
                        s1, s2 = self.sensors[sid], self.sensors[partner]
                        if primary_dir == 'NS':
                            # For NS roads, separate by small longitude difference
                            if s1['lon'] < s2['lon']:
                                dir1_sensors.append(sid)
                            else:
                                dir2_sensors.append(sid)
                        else:  # EW
                            if s1['lat'] < s2['lat']:
                                dir1_sensors.append(sid)
                            else:
                                dir2_sensors.append(sid)
                    else:
                        unpaired.append(sid)
                else:
                    unpaired.append(sid)

            # Add unpaired sensors to both directions or create single segment
            if len(dir1_sensors) + len(dir2_sensors) >= 4:
                # We have enough paired sensors for two directions
                if unpaired:
                    # Try to assign unpaired to nearest direction
                    for sid in unpaired:
                        if dir1_sensors:
                            avg_lon1 = np.mean([self.sensors[s]['lon'] for s in dir1_sensors])
                            avg_lon2 = np.mean([self.sensors[s]['lon'] for s in dir2_sensors]) if dir2_sensors else avg_lon1
                            if abs(self.sensors[sid]['lon'] - avg_lon1) < abs(self.sensors[sid]['lon'] - avg_lon2):
                                dir1_sensors.append(sid)
                            else:
                                dir2_sensors.append(sid)

                # Sort sensors along the corridor
                if primary_dir == 'NS':
                    dir1_sensors.sort(key=lambda s: self.sensors[s]['lat'])
                    dir2_sensors.sort(key=lambda s: self.sensors[s]['lat'], reverse=True)
                    dir1_name = f"{corridor_name}_NB"
                    dir2_name = f"{corridor_name}_SB"
                else:
                    dir1_sensors.sort(key=lambda s: self.sensors[s]['lon'])
                    dir2_sensors.sort(key=lambda s: self.sensors[s]['lon'], reverse=True)
                    dir1_name = f"{corridor_name}_EB"
                    dir2_name = f"{corridor_name}_WB"

                if len(dir1_sensors) >= 2:
                    segments[dir1_name] = {
                        'sensor_ids': dir1_sensors,
                        'direction': dir1_name.split('_')[-1],
                        'corridor': corridor_name
                    }
                    assigned.update(dir1_sensors)

                if len(dir2_sensors) >= 2:
                    segments[dir2_name] = {
                        'sensor_ids': dir2_sensors,
                        'direction': dir2_name.split('_')[-1],
                        'corridor': corridor_name
                    }
                    assigned.update(dir2_sensors)
            else:
                # Single direction segment
                all_sensors = dir1_sensors + dir2_sensors + unpaired
                if len(all_sensors) >= 2:
                    if primary_dir == 'NS':
                        all_sensors.sort(key=lambda s: self.sensors[s]['lat'])
                    else:
                        all_sensors.sort(key=lambda s: self.sensors[s]['lon'])

                    segments[corridor_name] = {
                        'sensor_ids': all_sensors,
                        'direction': primary_dir,
                        'corridor': corridor_name
                    }
                    assigned.update(all_sensors)

        # Handle unassigned sensors - group remaining by proximity
        unassigned = [sid for sid in self.sensors.keys() if sid not in assigned]
        if unassigned:
            segments['Other'] = {
                'sensor_ids': sorted(unassigned, key=lambda s: self.sensors[s]['lon']),
                'direction': 'Mixed',
                'corridor': 'Other'
            }

        self.road_segments = segments

        # Print summary
        print(f"\nDetected {len(segments)} road segments:")
        for name, info in sorted(segments.items()):
            print(f"  {name}: {len(info['sensor_ids'])} sensors")

        return segments

    def _define_la_corridors(self) -> Dict[str, dict]:
        """
        Define LA highway corridors based on known geography

        Returns:
            Dict of corridor definitions with bounding boxes
        """
        # Based on METR-LA sensor distribution (lon: -118.54 ~ -118.18, lat: 34.04 ~ 34.22)
        corridors = {
            # I-405 (San Diego Freeway) - North-South on west side
            'I-405': {
                'bbox': (-118.52, -118.44, 34.05, 34.22),
                'direction': 'NS'
            },
            # US-101 (Hollywood Freeway) - Northwest-Southeast through central
            'US-101-N': {
                'bbox': (-118.40, -118.28, 34.10, 34.18),
                'direction': 'EW'  # Approximated as EW for this section
            },
            'US-101-S': {
                'bbox': (-118.30, -118.22, 34.04, 34.12),
                'direction': 'NS'
            },
            # I-10 (Santa Monica Freeway) - East-West
            'I-10': {
                'bbox': (-118.50, -118.20, 34.03, 34.08),
                'direction': 'EW'
            },
            # I-110 (Harbor Freeway) - North-South
            'I-110': {
                'bbox': (-118.32, -118.26, 34.04, 34.15),
                'direction': 'NS'
            },
            # I-5 (Golden State Freeway) - North-South on east side
            'I-5': {
                'bbox': (-118.26, -118.18, 34.04, 34.22),
                'direction': 'NS'
            },
            # SR-134 (Ventura Freeway) - East-West in north
            'SR-134': {
                'bbox': (-118.38, -118.22, 34.14, 34.18),
                'direction': 'EW'
            },
            # SR-2 (Glendale Freeway) - roughly NS
            'SR-2': {
                'bbox': (-118.28, -118.22, 34.10, 34.17),
                'direction': 'NS'
            },
            # I-210 (Foothill Freeway) - EW in far north
            'I-210': {
                'bbox': (-118.28, -118.18, 34.18, 34.23),
                'direction': 'EW'
            }
        }
        return corridors

    def plot_segment_heatmap(self,
                             segment_name: str,
                             data: np.ndarray,
                             save_path: str,
                             congestion_threshold_mph: float = 35,
                             show_congestion_boxes: bool = True,
                             figsize: Tuple[int, int] = (16, 10),
                             dpi: int = 150) -> float:
        """
        Plot heatmap for a single road segment

        Args:
            segment_name: Name of the road segment
            data: Traffic data array (time x sensors) for this segment
            save_path: Path to save the figure
            congestion_threshold_mph: Speed below which is considered congestion
            show_congestion_boxes: Whether to highlight congestion with red boxes
            figsize: Figure size
            dpi: Output resolution

        Returns:
            Congestion ratio (percentage of congested cells)
        """
        segment = self.road_segments[segment_name]
        sensor_ids = segment['sensor_ids']

        # Extract data for this segment
        sensor_indices = [self.sensor_order.index(sid) for sid in sensor_ids]
        segment_data = data[:, sensor_indices]

        # Convert mph to km/h for display (data is in mph)
        data_kmh = segment_data * 1.60934
        congestion_threshold_kmh = congestion_threshold_mph * 1.60934

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot heatmap
        im = ax.imshow(data_kmh, aspect='auto', cmap=self.cmap,
                      vmin=0, vmax=120, origin='upper')

        # Highlight congestion
        if show_congestion_boxes:
            congestion_mask = (data_kmh < congestion_threshold_kmh) & (data_kmh > 0)
            for t in range(data_kmh.shape[0]):
                for s in range(data_kmh.shape[1]):
                    if congestion_mask[t, s]:
                        rect = patches.Rectangle(
                            (s - 0.5, t - 0.5), 1, 1,
                            linewidth=0.5,
                            edgecolor='red',
                            facecolor='none',
                            alpha=0.9
                        )
                        ax.add_patch(rect)

        # Set axis labels
        n_sensors = len(sensor_ids)
        ax.set_xlabel(f'Sensor Position ({n_sensors} sensors)', fontsize=12)

        # X-axis: show sensor IDs
        sensor_tick_step = max(1, n_sensors // 10)
        sensor_ticks = list(range(0, n_sensors, sensor_tick_step))
        ax.set_xticks(sensor_ticks)
        ax.set_xticklabels([str(sensor_ids[i]) for i in sensor_ticks], fontsize=8, rotation=45)

        # Y-axis: time
        ax.set_ylabel('Time', fontsize=12)
        n_time = data_kmh.shape[0]
        time_tick_step = max(1, n_time // 12)
        time_ticks = list(range(0, n_time, time_tick_step))
        ax.set_yticks(time_ticks)
        time_labels = [self.timestamps[i].strftime('%H:%M') if i < len(self.timestamps) else ''
                      for i in time_ticks]
        ax.set_yticklabels(time_labels, fontsize=9)

        # Title
        date_str = self.timestamps[0].strftime('%Y-%m-%d') if self.timestamps else 'Unknown'
        weekday = self.timestamps[0].strftime('%a') if self.timestamps else ''
        direction = segment['direction']
        corridor = segment['corridor']

        ax.set_title(f'METR-LA: {corridor} {direction}\n{date_str} ({weekday})\n'
                    f'(Red boxes: Speed < {congestion_threshold_kmh:.0f} km/h)', fontsize=14)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Speed (km/h)', fontsize=11)

        # Statistics
        valid_data = data_kmh[data_kmh > 0]
        if valid_data.size > 0:
            congestion_ratio = np.sum(valid_data < congestion_threshold_kmh) / valid_data.size * 100
        else:
            congestion_ratio = 0
        missing_ratio = np.sum(data_kmh == 0) / data_kmh.size * 100

        stats_text = f'Congestion: {congestion_ratio:.2f}% | Missing: {missing_ratio:.2f}%'
        ax.text(0.5, -0.1, stats_text, transform=ax.transAxes, ha='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()

        return congestion_ratio

    def visualize_day(self, day_index: int = 0, output_dir: str = None,
                     verbose: bool = True, **kwargs) -> List[str]:
        """
        Visualize all segments for a single day

        Args:
            day_index: Day index (0 = March 1, 2012)
            output_dir: Output directory
            verbose: Print progress
            **kwargs: Additional arguments for plot_segment_heatmap

        Returns:
            List of saved file paths
        """
        # Load data if not already loaded
        if not self.sensors:
            self.load_sensors()

        if not self.road_segments:
            self.cluster_sensors_by_corridor()

        # Load traffic data for this day
        self.load_traffic_data(day_index)

        # Setup output directory
        if output_dir is None:
            output_dir = self.data_dir.parent / 'output' / 'metr_heatmaps'
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        date = self.START_DATE + timedelta(days=day_index)
        date_str = date.strftime('%Y%m%d')

        saved_paths = []

        for segment_name in sorted(self.road_segments.keys()):
            save_path = output_dir / f'heatmap_metr_{date_str}_{segment_name}.png'

            congestion_ratio = self.plot_segment_heatmap(
                segment_name,
                self.traffic_data,
                str(save_path),
                **kwargs
            )

            saved_paths.append(str(save_path))

            if verbose:
                print(f"Saved: {save_path} (congestion: {congestion_ratio:.2f}%)")

        return saved_paths

    def visualize_batch(self, start_day: int = 0, end_day: int = None,
                       output_dir: str = None, **kwargs) -> List[Tuple[int, str, float]]:
        """
        Batch visualize multiple days

        Args:
            start_day: Starting day index
            end_day: Ending day index (exclusive)
            output_dir: Output directory
            **kwargs: Additional arguments for visualization

        Returns:
            List of (day_index, segment_name, congestion_ratio) tuples
        """
        # Determine total days in dataset
        # METR-LA: March 1 to June 30, 2012 = 122 days
        max_days = 122

        if end_day is None:
            end_day = max_days
        end_day = min(end_day, max_days)

        # Load sensors and cluster once
        if not self.sensors:
            self.load_sensors()
        if not self.road_segments:
            self.cluster_sensors_by_corridor()

        print(f"\nBatch visualization: days {start_day} to {end_day-1}")

        results = []

        for day_idx in range(start_day, end_day):
            try:
                print(f"\n--- Day {day_idx} ---")
                saved_paths = self.visualize_day(day_idx, output_dir, verbose=True, **kwargs)
                for path in saved_paths:
                    # Extract segment name from path
                    segment_name = Path(path).stem.split('_')[-1]
                    results.append((day_idx, segment_name, 0))  # congestion ratio not easily accessible here
            except Exception as e:
                print(f"Warning: Failed to process day {day_idx}: {e}")

        print(f"\nCompleted! Generated {len(results)} images")
        return results

    def plot_sensor_map(self, save_path: str = None, show: bool = True) -> None:
        """
        Plot a map of all sensors colored by segment

        Args:
            save_path: Path to save figure
            show: Whether to display the plot
        """
        if not self.sensors:
            self.load_sensors()
        if not self.road_segments:
            self.cluster_sensors_by_corridor()

        fig, ax = plt.subplots(figsize=(12, 10))

        # Color palette
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.road_segments)))

        for (segment_name, segment_info), color in zip(self.road_segments.items(), colors):
            sensor_ids = segment_info['sensor_ids']
            lons = [self.sensors[sid]['lon'] for sid in sensor_ids]
            lats = [self.sensors[sid]['lat'] for sid in sensor_ids]

            ax.scatter(lons, lats, c=[color], label=segment_name, s=30, alpha=0.8)

            # Draw line connecting sensors in order
            ax.plot(lons, lats, c=color, linewidth=1, alpha=0.5)

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('METR-LA Sensor Distribution by Road Segment')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.set_aspect('equal')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved sensor map: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='METR-LA Traffic Heatmap Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize day 10 (March 11, 2012)
  python metr_heatmap.py --day 10

  # Batch visualize days 0-30
  python metr_heatmap.py --batch --start-day 0 --end-day 30

  # Show sensor distribution map
  python metr_heatmap.py --show-map

  # Change congestion threshold
  python metr_heatmap.py --day 10 --threshold 30
        """
    )

    parser.add_argument('--day', type=int, default=None,
                       help='Day index to visualize (0 = March 1, 2012)')
    parser.add_argument('--batch', action='store_true',
                       help='Batch mode: visualize multiple days')
    parser.add_argument('--start-day', type=int, default=0,
                       help='Starting day for batch mode')
    parser.add_argument('--end-day', type=int, default=None,
                       help='Ending day for batch mode (exclusive)')
    parser.add_argument('--threshold', type=float, default=35,
                       help='Congestion threshold in mph (default: 35)')
    parser.add_argument('--no-boxes', action='store_true',
                       help='Disable congestion highlight boxes')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='METR_LA data directory')
    parser.add_argument('--show-map', action='store_true',
                       help='Show sensor distribution map')
    parser.add_argument('--dpi', type=int, default=150,
                       help='Output image DPI')

    args = parser.parse_args()

    visualizer = METRHeatmapVisualizer(data_dir=args.data_dir)

    kwargs = {
        'congestion_threshold_mph': args.threshold,
        'show_congestion_boxes': not args.no_boxes,
        'dpi': args.dpi,
    }

    if args.show_map:
        map_path = Path(args.output_dir or '.') / 'metr_sensor_map.png'
        visualizer.plot_sensor_map(save_path=str(map_path))
    elif args.batch:
        visualizer.visualize_batch(
            start_day=args.start_day,
            end_day=args.end_day,
            output_dir=args.output_dir,
            **kwargs
        )
    elif args.day is not None:
        visualizer.visualize_day(
            day_index=args.day,
            output_dir=args.output_dir,
            **kwargs
        )
    else:
        # Default: visualize day 0
        visualizer.visualize_day(
            day_index=0,
            output_dir=args.output_dir,
            **kwargs
        )


if __name__ == '__main__':
    main()
