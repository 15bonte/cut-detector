# Notes on the way `LapTrack.predict` works #

## `predict` VS `predict_dataframe`
LapTrack.predict is basically the real entry point.

`predict_dataframe` is just a wrapper around `predict`.
It first modifies the dataframe then converts it to the list of points.
Then calls `predict`.
Finally it turns the networkx DiGraph into a dataframe (with more or less
columns depending on the parameters).

## Call analysis

### `predict`
- Validation of Input
- known connected_edges handling
- Simple linking 2 frames by 2 frames with `_predict_link`
- Handling gap/split/merge: `_predict_gap_split_merge`
- Creating DiGraph from Track Tree

#### `_predict_links`
- Initialises Graph
- (handle known connected edges)
- Parallel vs Serial Execution of `_predict_link_single_frame`
- Add the obtained nodes
- Add the known nodes

##### `_predict_link_single_frame`
- > Compute distance matrix with metric
- > Compute an index mask: distance < max_distance 
- Distance matrix is turned into cost matrix 
  (small one, without new points (gap/split/merge)) and computed
- Edges are returned

#### `_predict_gap_split_merge`
- Executed only if gap/split/merge is enabled
- Creating segment_df: matrix of segments points / first frame / last frame / coords
- Computing distance matrix from segment_df: `_get_gap_closing_matrix`
- (Handling splitting/merging)
- Solving everything in `_link_gap_split_merge_from_matrix`

##### `_get_gap_closing_matrix`
If gap closing is allowed:
- Parallel Vs Serial Execution of `to_gap_closing_candidates` 
  on segments_df, modifying segments_df
- Making a cost matrix from the results, type conversions...

###### `to_gap_closing_candidates`
Generating a distance matrix for allowed points: neither past/present frames (only future frames) AND withing the max frame distance allowed
- Filtering out possibilities (see above)
- (handling force-start edges)
- > Compute gap closing dist matrix based on a metric
- > a mask comparing it to threshold value
- filtering out gap closing matrix based on mask

##### `_link_gap_split_merge_from_matrix`
- building full cost matrix: `build_segment_cost_matrix`
- lap_optimization
- Filling tree from it (structure gives the answer)

###### `build_segment_cost_matrix`
- Combining gap/split/merge matrices (all but upper-left)
- Computing percentiles
- Computing upper-left from ones multiples by percentile-based and cost-based values
- Adding epsilon (to remove zeroes ?)
- return



