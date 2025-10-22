<#
.SYNOPSIS
DINOv3 Weight Downloader with menu & progress bar
#>

# --- CONFIGURE WEIGHTS ---
# Format: "DisplayName|FileName|URL"
$lvdVitModels = @(
    "ViT-S/16 distilled|dinov3_vits16_lvd.pth|https://dinov3.llamameta.net/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZ3VuNzFqa3gzdzBlMno1emF3dWVxYm9vIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjEzMzE4ODZ9fX1dfQ__&Signature=g%7EWQAXKWzhW7sRw0r6IHUscpFChdS43Zy2ydjOzQ-fpTFxRt9Eh4i91JMulajjGKHfTAhqA9zHrb-Ra1qa0ByWmtSYxwsqDExEw1oWqb14FMXlxaP3b9s5PRw1eabyXs0g-fQGFzAxqasTjv56uw01nP%7EuzgkTEupK4tbZJCo7OVE-qpjDpQQ0Ilj%7EOYWqt9oyMAmDTb-ruQh5kmFdh%7EGrmB5LlO0xUUCqtuAhpmYWfjYbsBgBwdLFFhfI3En3smunTHoZ73Y8jzkdw%7EmNVFj9proYsvDluidZlgsWI2PokoM51qgvl0zlzQr9YIwrY%7EVfdOqJ0ZW1ZlO%7ET5Epi8YA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=844403288241374",
    "ViT-S+/16 distilled|dinov3_vits16plus_lvd.pth|https://dinov3.llamameta.net/dinov3_vits16plus/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZ3VuNzFqa3gzdzBlMno1emF3dWVxYm9vIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjEzMzE4ODZ9fX1dfQ__&Signature=g%7EWQAXKWzhW7sRw0r6IHUscpFChdS43Zy2ydjOzQ-fpTFxRt9Eh4i91JMulajjGKHfTAhqA9zHrb-Ra1qa0ByWmtSYxwsqDExEw1oWqb14FMXlxaP3b9s5PRw1eabyXs0g-fQGFzAxqasTjv56uw01nP%7EuzgkTEupK4tbZJCo7OVE-qpjDpQQ0Ilj%7EOYWqt9oyMAmDTb-ruQh5kmFdh%7EGrmB5LlO0xUUCqtuAhpmYWfjYbsBgBwdLFFhfI3En3smunTHoZ73Y8jzkdw%7EmNVFj9proYsvDluidZlgsWI2PokoM51qgvl0zlzQr9YIwrY%7EVfdOqJ0ZW1ZlO%7ET5Epi8YA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=844403288241374",
    "ViT-B/16 distilled|dinov3_vitb16_lvd.pth|https://dinov3.llamameta.net/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZ3VuNzFqa3gzdzBlMno1emF3dWVxYm9vIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjEzMzE4ODZ9fX1dfQ__&Signature=g%7EWQAXKWzhW7sRw0r6IHUscpFChdS43Zy2ydjOzQ-fpTFxRt9Eh4i91JMulajjGKHfTAhqA9zHrb-Ra1qa0ByWmtSYxwsqDExEw1oWqb14FMXlxaP3b9s5PRw1eabyXs0g-fQGFzAxqasTjv56uw01nP%7EuzgkTEupK4tbZJCo7OVE-qpjDpQQ0Ilj%7EOYWqt9oyMAmDTb-ruQh5kmFdh%7EGrmB5LlO0xUUCqtuAhpmYWfjYbsBgBwdLFFhfI3En3smunTHoZ73Y8jzkdw%7EmNVFj9proYsvDluidZlgsWI2PokoM51qgvl0zlzQr9YIwrY%7EVfdOqJ0ZW1ZlO%7ET5Epi8YA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=844403288241374",
    "ViT-L/16 distilled|dinov3_vitl16_lvd.pth|https://dinov3.llamameta.net/dinov3_vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZ3VuNzFqa3gzdzBlMno1emF3dWVxYm9vIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjEzMzE4ODZ9fX1dfQ__&Signature=g%7EWQAXKWzhW7sRw0r6IHUscpFChdS43Zy2ydjOzQ-fpTFxRt9Eh4i91JMulajjGKHfTAhqA9zHrb-Ra1qa0ByWmtSYxwsqDExEw1oWqb14FMXlxaP3b9s5PRw1eabyXs0g-fQGFzAxqasTjv56uw01nP%7EuzgkTEupK4tbZJCo7OVE-qpjDpQQ0Ilj%7EOYWqt9oyMAmDTb-ruQh5kmFdh%7EGrmB5LlO0xUUCqtuAhpmYWfjYbsBgBwdLFFhfI3En3smunTHoZ73Y8jzkdw%7EmNVFj9proYsvDluidZlgsWI2PokoM51qgvl0zlzQr9YIwrY%7EVfdOqJ0ZW1ZlO%7ET5Epi8YA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=844403288241374",
    "ViT-H+/16 distilled|dinov3_vith16plus_lvd.pth|https://dinov3.llamameta.net/dinov3_vith16plus/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZ3VuNzFqa3gzdzBlMno1emF3dWVxYm9vIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjEzMzE4ODZ9fX1dfQ__&Signature=g%7EWQAXKWzhW7sRw0r6IHUscpFChdS43Zy2ydjOzQ-fpTFxRt9Eh4i91JMulajjGKHfTAhqA9zHrb-Ra1qa0ByWmtSYxwsqDExEw1oWqb14FMXlxaP3b9s5PRw1eabyXs0g-fQGFzAxqasTjv56uw01nP%7EuzgkTEupK4tbZJCo7OVE-qpjDpQQ0Ilj%7EOYWqt9oyMAmDTb-ruQh5kmFdh%7EGrmB5LlO0xUUCqtuAhpmYWfjYbsBgBwdLFFhfI3En3smunTHoZ73Y8jzkdw%7EmNVFj9proYsvDluidZlgsWI2PokoM51qgvl0zlzQr9YIwrY%7EVfdOqJ0ZW1ZlO%7ET5Epi8YA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=844403288241374",
    "ViT-7B/16|dinov3_vit7b16_lvd.pth|https://dinov3.llamameta.net/dinov3_vit7b16/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZ3VuNzFqa3gzdzBlMno1emF3dWVxYm9vIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjEzMzE4ODZ9fX1dfQ__&Signature=g%7EWQAXKWzhW7sRw0r6IHUscpFChdS43Zy2ydjOzQ-fpTFxRt9Eh4i91JMulajjGKHfTAhqA9zHrb-Ra1qa0ByWmtSYxwsqDExEw1oWqb14FMXlxaP3b9s5PRw1eabyXs0g-fQGFzAxqasTjv56uw01nP%7EuzgkTEupK4tbZJCo7OVE-qpjDpQQ0Ilj%7EOYWqt9oyMAmDTb-ruQh5kmFdh%7EGrmB5LlO0xUUCqtuAhpmYWfjYbsBgBwdLFFhfI3En3smunTHoZ73Y8jzkdw%7EmNVFj9proYsvDluidZlgsWI2PokoM51qgvl0zlzQr9YIwrY%7EVfdOqJ0ZW1ZlO%7ET5Epi8YA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=844403288241374"
)

$lvdConvnextModels = @(
    "ConvNeXt Tiny|dinov3_convnext_tiny_lvd.pth|https://dinov3.llamameta.net/dinov3_convnext_tiny/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZ3VuNzFqa3gzdzBlMno1emF3dWVxYm9vIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjEzMzE4ODZ9fX1dfQ__&Signature=g%7EWQAXKWzhW7sRw0r6IHUscpFChdS43Zy2ydjOzQ-fpTFxRt9Eh4i91JMulajjGKHfTAhqA9zHrb-Ra1qa0ByWmtSYxwsqDExEw1oWqb14FMXlxaP3b9s5PRw1eabyXs0g-fQGFzAxqasTjv56uw01nP%7EuzgkTEupK4tbZJCo7OVE-qpjDpQQ0Ilj%7EOYWqt9oyMAmDTb-ruQh5kmFdh%7EGrmB5LlO0xUUCqtuAhpmYWfjYbsBgBwdLFFhfI3En3smunTHoZ73Y8jzkdw%7EmNVFj9proYsvDluidZlgsWI2PokoM51qgvl0zlzQr9YIwrY%7EVfdOqJ0ZW1ZlO%7ET5Epi8YA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=844403288241374",
    "ConvNeXt Small|dinov3_convnext_small_lvd.pth|https://dinov3.llamameta.net/dinov3_convnext_small/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZ3VuNzFqa3gzdzBlMno1emF3dWVxYm9vIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjEzMzE4ODZ9fX1dfQ__&Signature=g%7EWQAXKWzhW7sRw0r6IHUscpFChdS43Zy2ydjOzQ-fpTFxRt9Eh4i91JMulajjGKHfTAhqA9zHrb-Ra1qa0ByWmtSYxwsqDExEw1oWqb14FMXlxaP3b9s5PRw1eabyXs0g-fQGFzAxqasTjv56uw01nP%7EuzgkTEupK4tbZJCo7OVE-qpjDpQQ0Ilj%7EOYWqt9oyMAmDTb-ruQh5kmFdh%7EGrmB5LlO0xUUCqtuAhpmYWfjYbsBgBwdLFFhfI3En3smunTHoZ73Y8jzkdw%7EmNVFj9proYsvDluidZlgsWI2PokoM51qgvl0zlzQr9YIwrY%7EVfdOqJ0ZW1ZlO%7ET5Epi8YA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=844403288241374",
    "ConvNeXt Base|dinov3_convnext_base_lvd.pth|https://dinov3.llamameta.net/dinov3_convnext_base/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZ3VuNzFqa3gzdzBlMno1emF3dWVxYm9vIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjEzMzE4ODZ9fX1dfQ__&Signature=g%7EWQAXKWzhW7sRw0r6IHUscpFChdS43Zy2ydjOzQ-fpTFxRt9Eh4i91JMulajjGKHfTAhqA9zHrb-Ra1qa0ByWmtSYxwsqDExEw1oWqb14FMXlxaP3b9s5PRw1eabyXs0g-fQGFzAxqasTjv56uw01nP%7EuzgkTEupK4tbZJCo7OVE-qpjDpQQ0Ilj%7EOYWqt9oyMAmDTb-ruQh5kmFdh%7EGrmB5LlO0xUUCqtuAhpmYWfjYbsBgBwdLFFhfI3En3smunTHoZ73Y8jzkdw%7EmNVFj9proYsvDluidZlgsWI2PokoM51qgvl0zlzQr9YIwrY%7EVfdOqJ0ZW1ZlO%7ET5Epi8YA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=844403288241374",
    "ConvNeXt Large|dinov3_convnext_large_lvd.pth|https://dinov3.llamameta.net/dinov3_convnext_large/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZ3VuNzFqa3gzdzBlMno1emF3dWVxYm9vIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjEzMzE4ODZ9fX1dfQ__&Signature=g%7EWQAXKWzhW7sRw0r6IHUscpFChdS43Zy2ydjOzQ-fpTFxRt9Eh4i91JMulajjGKHfTAhqA9zHrb-Ra1qa0ByWmtSYxwsqDExEw1oWqb14FMXlxaP3b9s5PRw1eabyXs0g-fQGFzAxqasTjv56uw01nP%7EuzgkTEupK4tbZJCo7OVE-qpjDpQQ0Ilj%7EOYWqt9oyMAmDTb-ruQh5kmFdh%7EGrmB5LlO0xUUCqtuAhpmYWfjYbsBgBwdLFFhfI3En3smunTHoZ73Y8jzkdw%7EmNVFj9proYsvDluidZlgsWI2PokoM51qgvl0zlzQr9YIwrY%7EVfdOqJ0ZW1ZlO%7ET5Epi8YA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=844403288241374"
)

$satVitModels = @(
    "ViT-L/16 distilled|dinov3_vitl16_sat.pth|https://dinov3.llamameta.net/dinov3_vitl16/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZ3VuNzFqa3gzdzBlMno1emF3dWVxYm9vIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjEzMzE4ODZ9fX1dfQ__&Signature=g%7EWQAXKWzhW7sRw0r6IHUscpFChdS43Zy2ydjOzQ-fpTFxRt9Eh4i91JMulajjGKHfTAhqA9zHrb-Ra1qa0ByWmtSYxwsqDExEw1oWqb14FMXlxaP3b9s5PRw1eabyXs0g-fQGFzAxqasTjv56uw01nP%7EuzgkTEupK4tbZJCo7OVE-qpjDpQQ0Ilj%7EOYWqt9oyMAmDTb-ruQh5kmFdh%7EGrmB5LlO0xUUCqtuAhpmYWfjYbsBgBwdLFFhfI3En3smunTHoZ73Y8jzkdw%7EmNVFj9proYsvDluidZlgsWI2PokoM51qgvl0zlzQr9YIwrY%7EVfdOqJ0ZW1ZlO%7ET5Epi8YA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=844403288241374",
    "ViT-7B/16|dinov3_vit7b16_sat.pth|https://dinov3.llamameta.net/dinov3_vit7b16/dinov3_vit7b16_pretrain_sat493m-a6675841.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZ3VuNzFqa3gzdzBlMno1emF3dWVxYm9vIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjEzMzE4ODZ9fX1dfQ__&Signature=g%7EWQAXKWzhW7sRw0r6IHUscpFChdS43Zy2ydjOzQ-fpTFxRt9Eh4i91JMulajjGKHfTAhqA9zHrb-Ra1qa0ByWmtSYxwsqDExEw1oWqb14FMXlxaP3b9s5PRw1eabyXs0g-fQGFzAxqasTjv56uw01nP%7EuzgkTEupK4tbZJCo7OVE-qpjDpQQ0Ilj%7EOYWqt9oyMAmDTb-ruQh5kmFdh%7EGrmB5LlO0xUUCqtuAhpmYWfjYbsBgBwdLFFhfI3En3smunTHoZ73Y8jzkdw%7EmNVFj9proYsvDluidZlgsWI2PokoM51qgvl0zlzQr9YIwrY%7EVfdOqJ0ZW1ZlO%7ET5Epi8YA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=844403288241374"
)

$adapterModels = @(
    "Classifier (ImageNet)|adapter_vit7b_classifier_inet.pth|https://dinov3.llamameta.net/dinov3_vit7b16/dinov3_vit7b16_imagenet1k_linear_head-90d8ed92.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZ3VuNzFqa3gzdzBlMno1emF3dWVxYm9vIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjEzMzE4ODZ9fX1dfQ__&Signature=g%7EWQAXKWzhW7sRw0r6IHUscpFChdS43Zy2ydjOzQ-fpTFxRt9Eh4i91JMulajjGKHfTAhqA9zHrb-Ra1qa0ByWmtSYxwsqDExEw1oWqb14FMXlxaP3b9s5PRw1eabyXs0g-fQGFzAxqasTjv56uw01nP%7EuzgkTEupK4tbZJCo7OVE-qpjDpQQ0Ilj%7EOYWqt9oyMAmDTb-ruQh5kmFdh%7EGrmB5LlO0xUUCqtuAhpmYWfjYbsBgBwdLFFhfI3En3smunTHoZ73Y8jzkdw%7EmNVFj9proYsvDluidZlgsWI2PokoM51qgvl0zlzQr9YIwrY%7EVfdOqJ0ZW1ZlO%7ET5Epi8YA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=844403288241374",
    "Depther (SYNTHMIX)|adapter_vit7b_depther_synthmix.pth|https://dinov3.llamameta.net/dinov3_vit7b16/dinov3_vit7b16_synthmix_dpt_head-02040be1.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZ3VuNzFqa3gzdzBlMno1emF3dWVxYm9vIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjEzMzE4ODZ9fX1dfQ__&Signature=g%7EWQAXKWzhW7sRw0r6IHUscpFChdS43Zy2ydjOzQ-fpTFxRt9Eh4i91JMulajjGKHfTAhqA9zHrb-Ra1qa0ByWmtSYxwsqDExEw1oWqb14FMXlxaP3b9s5PRw1eabyXs0g-fQGFzAxqasTjv56uw01nP%7EuzgkTEupK4tbZJCo7OVE-qpjDpQQ0Ilj%7EOYWqt9oyMAmDTb-ruQh5kmFdh%7EGrmB5LlO0xUUCqtuAhpmYWfjYbsBgBwdLFFhfI3En3smunTHoZ73Y8jzkdw%7EmNVFj9proYsvDluidZlgsWI2PokoM51qgvl0zlzQr9YIwrY%7EVfdOqJ0ZW1ZlO%7ET5Epi8YA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=844403288241374",
    "Detector (COCO2017)|adapter_vit7b_detector_coco.pth|https://dinov3.llamameta.net/dinov3_vit7b16/dinov3_vit7b16_coco_detr_head-b0235ff7.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZ3VuNzFqa3gzdzBlMno1emF3dWVxYm9vIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjEzMzE4ODZ9fX1dfQ__&Signature=g%7EWQAXKWzhW7sRw0r6IHUscpFChdS43Zy2ydjOzQ-fpTFxRt9Eh4i91JMulajjGKHfTAhqA9zHrb-Ra1qa0ByWmtSYxwsqDExEw1oWqb14FMXlxaP3b9s5PRw1eabyXs0g-fQGFzAxqasTjv56uw01nP%7EuzgkTEupK4tbZJCo7OVE-qpjDpQQ0Ilj%7EOYWqt9oyMAmDTb-ruQh5kmFdh%7EGrmB5LlO0xUUCqtuAhpmYWfjYbsBgBwdLFFhfI3En3smunTHoZ73Y8jzkdw%7EmNVFj9proYsvDluidZlgsWI2PokoM51qgvl0zlzQr9YIwrY%7EVfdOqJ0ZW1ZlO%7ET5Epi8YA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=844403288241374",
    "Segmentor (ADE20K)|adapter_vit7b_segmentor_ade20k.pth|https://dinov3.llamameta.net/dinov3_vit7b16/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZ3VuNzFqa3gzdzBlMno1emF3dWVxYm9vIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjEzMzE4ODZ9fX1dfQ__&Signature=g%7EWQAXKWzhW7sRw0r6IHUscpFChdS43Zy2ydjOzQ-fpTFxRt9Eh4i91JMulajjGKHfTAhqA9zHrb-Ra1qa0ByWmtSYxwsqDExEw1oWqb14FMXlxaP3b9s5PRw1eabyXs0g-fQGFzAxqasTjv56uw01nP%7EuzgkTEupK4tbZJCo7OVE-qpjDpQQ0Ilj%7EOYWqt9oyMAmDTb-ruQh5kmFdh%7EGrmB5LlO0xUUCqtuAhpmYWfjYbsBgBwdLFFhfI3En3smunTHoZ73Y8jzkdw%7EmNVFj9proYsvDluidZlgsWI2PokoM51qgvl0zlzQr9YIwrY%7EVfdOqJ0ZW1ZlO%7ET5Epi8YA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=844403288241374",
    "DINO.txt (ViT-L/16)|adapter_vitl16_dinotxt.pth|https://dinov3.llamameta.net/dinov3_vitl16/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZ3VuNzFqa3gzdzBlMno1emF3dWVxYm9vIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjEzMzE4ODZ9fX1dfQ__&Signature=g%7EWQAXKWzhW7sRw0r6IHUscpFChdS43Zy2ydjOzQ-fpTFxRt9Eh4i91JMulajjGKHfTAhqA9zHrb-Ra1qa0ByWmtSYxwsqDExEw1oWqb14FMXlxaP3b9s5PRw1eabyXs0g-fQGFzAxqasTjv56uw01nP%7EuzgkTEupK4tbZJCo7OVE-qpjDpQQ0Ilj%7EOYWqt9oyMAmDTb-ruQh5kmFdh%7EGrmB5LlO0xUUCqtuAhpmYWfjYbsBgBwdLFFhfI3En3smunTHoZ73Y8jzkdw%7EmNVFj9proYsvDluidZlgsWI2PokoM51qgvl0zlzQr9YIwrY%7EVfdOqJ0ZW1ZlO%7ET5Epi8YA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=844403288241374"
)

# --- FUNCTIONS ---

function Show-Menu($title, $options) {
    Write-Host "`n$title`n"
    for ($i=0; $i -lt $options.Count; $i++) {
        Write-Host "[$($i+1)] $($options[$i])"
    }
    Write-Host "[0] Back"

    do {
        $selection = Read-Host "Enter choice number"
        if ($selection -eq '0') { return -1 }
        elseif (($selection -as [int]) -ge 1 -and ($selection -as [int]) -le $options.Count) {
            return ($selection - 1)
        } else {
            Write-Host "Invalid selection. Try again."
        }
    } while ($true)
}

function Download-File($url, $output) {
    if ($url -eq "PASTE_URL_HERE") {
        Write-Warning "URL not set for $output"
        return
    }

    Write-Host "Downloading $output ..."
    
    $wc = New-Object System.Net.WebClient

    $progress = {
        param($sender, $e)
        $barLength = 50
        $pct = [int]($e.ProgressPercentage)
        $filled = [int]($pct * $barLength / 100)
        $empty = $barLength - $filled
        $bar = ("█" * $filled) + ("-" * $empty)
        Write-Host -NoNewline "`r[$bar] $pct% "
    }

    $wc.DownloadProgressChanged += $progress
    $wc.DownloadFileAsync($url, $output)

    # Wait for completion
    while ($wc.IsBusy) { Start-Sleep -Milliseconds 200 }
    Write-Host "`n✅ Download completed: $output"
}

function Download-Menu($models) {
    $names = $models | ForEach-Object { ($_ -split '\|')[0] }
    $idx = Show-Menu "Select model to download:" $names
    if ($idx -ge 0) {
        $entry = $models[$idx] -split '\|'
        $filename = $entry[1]
        $url = $entry[2]
        Download-File $url $filename
    }
}

function Backbone-Menu() {
    $options = @("ViT", "ConvNeXt")
    $idx = Show-Menu "Choose architecture:" $options
    switch ($idx) {
        0 {
            $datasetOptions = @("LVD-1689M", "SAT-493M")
            $dsIdx = Show-Menu "Select pre-training dataset:" $datasetOptions
            switch ($dsIdx) {
                0 { Download-Menu $lvdVitModels }
                1 { Download-Menu $satVitModels }
            }
        }
        1 { Download-Menu $lvdConvnextModels }
    }
}

function Adapter-Menu() {
    Download-Menu $adapterModels
}

# --- MAIN LOOP ---
do {
    $mainOptions = @("Download Backbone", "Download Adapters", "Quit")
    $mainIdx = Show-Menu "DINOv3 Weight Downloader" $mainOptions
    switch ($mainIdx) {
        0 { Backbone-Menu }
        1 { Adapter-Menu }
        2 { break }
        default { break }
    }
} while ($true)

Write-Host "Exiting."
