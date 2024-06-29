import easy_mindspore

def slice_helper(slice_spec):
    if not isinstance(slice_spec, (list, tuple)):
        slice_spec = [slice_spec]

    begin, end, strides = (), (), ()
    index = 0

    new_axis_mask, shrink_axis_mask = 0, 0
    begin_mask, end_mask = 0, 0
    ellipsis_mask = 0

    for s in slice_spec:
        if isinstance(s, slice):
            if s.start is not None:
                begin += (s.start,)
            else:
                begin += (0,)
                begin_mask |= (1 << index)
            
            if s.stop is not None:
                end += (s.stop,)
            else:
                end += (0,)
                end_mask |= (1 << index)

            if s.step is not None:
                strides += (s.step,)
            else:
                strides += (1,)
        elif s is Ellipsis:
            begin += (0,)
            end += (0,)
            strides += (1,)
            ellipsis_mask |= (1 << index)
        elif s is None:
            # begin += (0,)
            # end += (0,)
            # strides += (0,)
            new_axis_mask |= (1 << index)
        else:
            begin += (s,)
            end += (s + 1,)
            strides += (1,)
            shrink_axis_mask |= (1 << index)
        index += 1

    return begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask

def _get_unfold_indices(input_shape, dimension, size, step):
    if dimension < 0:
        dimension += len(input_shape)
    indices = []
    for i in range(0, input_shape[dimension] - size + 1, step):
        indices.append(list(range(i, i + size)))

    indices = easy_mindspore.tensor(indices)
    return indices, dimension
