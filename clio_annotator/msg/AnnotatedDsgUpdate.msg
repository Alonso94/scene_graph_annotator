# AnnotatedDsgUpdate.msg
Header header

# Original DSG update from Hydra/Clio
uint8[] layer_contents  # serialized nodes that are active
uint64[] deleted_nodes  # node ids that were deleted
uint64[] deleted_edges  # node ids for edges that were deleted
bool full_update       # whether or not the message contains the entire scene graph
int64 sequence_number  # update index

# Annotations added by AEG-LLM
Annotation[] annotations  # Per-node/edge annotations (link by ID)