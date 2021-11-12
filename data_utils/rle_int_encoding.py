def rle_to_matrix(rle_enc, width):
    int_rle = []
    temp_rle = []
    remaining = width
    for idx, el in enumerate(rle_enc):
        if el <= remaining:
            temp_rle.append(el)
            remaining -= el
            if remaining == 0:
                int_rle.append(temp_rle)
                if idx % 2 == 0:  # EMPTY / DISPARI
                    temp_rle = [0]  # We are assured that the next item will be a mask!!
                else:  # MASK / PARI
                    temp_rle = []
                remaining = width
        else:
            while remaining <= el:
                temp_rle.append(remaining)
                int_rle.append(temp_rle)
                if idx % 2 == 0:  # EMPTY / DISPARI
                    temp_rle = []
                else:  # MASK / PARI
                    temp_rle = [0]
                el -= remaining
                remaining = width

            if idx % 2 == 0:
                temp_rle.append(el)
                remaining -= el
            else:
                if el == 0:
                    temp_rle = []
                else:
                    temp_rle = [0, el]

    return int_rle


def scale_up_rle_enc(rle_enc, width, scale):
    int_rle = rle_to_matrix(rle_enc, width)
    new_mask = []
    for line in int_rle:
        line = [i * scale for i in line]
        for _ in range(scale):
            new_mask.append(line)
    scaled_rle = reconstruct_int_rle(new_mask)
    return scaled_rle
    # return new_mask


def reconstruct_int_rle(int_rle):
    new_rle_enc = []
    empty = 0
    mask = 0
    first_line = True
    for line in int_rle:
        for idx in range(len(line)):
            if idx % 2 == 0:  # odd position, empty
                if mask > 0 and line[idx] > 0:
                    new_rle_enc.append(mask)
                    mask = 0
                empty += line[idx]
            else:  # even position, mask
                if first_line and empty == 0:
                    new_rle_enc.append(empty)
                    first_line = False
                if empty > 0 and line[idx] > 0:
                    new_rle_enc.append(empty)
                    empty = 0
                mask += line[idx]
    # Add last empty if last empty is without masks
    if idx % 2 == 0:
        if empty > 0:
            new_rle_enc.append(empty)
    else:
        if mask > 0:
            new_rle_enc.append(mask)
    return new_rle_enc


def test_rle_to_matrix(rle_enc, width):
    int_rle = rle_to_matrix(rle_enc, width)
    re_rle_enc = reconstruct_int_rle(int_rle)
    assert rle_enc == re_rle_enc
    print(f"{rle_enc} is ok!")
