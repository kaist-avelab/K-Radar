#pragma once
#include <torch/extension.h>

#define MAX_NUM_VERT_IDX 9

at::Tensor sort_vertices(at::Tensor vertices, at::Tensor mask, at::Tensor num_valid);