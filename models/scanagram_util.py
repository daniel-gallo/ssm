import jax
import jax.numpy as jnp
import scanagram
from jax import lax


def pooled_block_scanagram_rule(block):
    def rule(scan_info, x):
        b, l, c = x.shape
        assert scan_info.axis == 1
        if scan_info.prefill is not None:
            raise NotImplementedError
        down_pool_carry = jnp.zeros((b, block.pool_temporal - 1, c))
        example_down_pooled_xs = jax.ShapeDtypeStruct(
            (l // block.pool_temporal, b, c * block.pool_feature), "float32"
        )

        def inner_layer(x):
            return jnp.moveaxis(
                block.inner_layer(
                    jnp.moveaxis(x, 0, 1), training=False, sampling=True
                ),
                1,
                0,
            )

        inner_body, inner_carry = scanagram.as_scan(
            inner_layer, example_down_pooled_xs
        )
        up_pool_carry = jnp.zeros((b, block.pool_temporal - 1, c))

        def body_fn(carry, x):
            t, down_pool_carry, inner_carry, up_pool_carry = carry
            x = jnp.expand_dims(x, 1)
            down_pool_input = jnp.concatenate([down_pool_carry, x], 1)
            down_pool_carry = down_pool_input[:, 1:]

            def do_inner():
                down_pool_output = block.down_pool(down_pool_input, True)
                down_pool_output = jnp.squeeze(down_pool_output, 1)
                inner_carry_, inner_output = inner_body(
                    inner_carry, down_pool_output
                )
                up_pool_input = jnp.expand_dims(inner_output, 1)
                up_pool_output = block.up_pool(up_pool_input)
                output, up_pool_carry = jnp.split(up_pool_output, [1], 1)
                return output, inner_carry_, up_pool_carry

            def dont_inner():
                return (
                    lax.dynamic_index_in_dim(
                        up_pool_carry, (t % block.pool_temporal) - 1, 1
                    ),
                    inner_carry,
                    up_pool_carry,
                )

            output, inner_carry, up_pool_carry = lax.cond(
                t % block.pool_temporal, dont_inner, do_inner
            )
            output = jnp.squeeze(output, 1)
            return (t + 1, down_pool_carry, inner_carry, up_pool_carry), output

        return (
            scanagram.ScanInfo(1, None),
            body_fn,
            (0, down_pool_carry, inner_carry, up_pool_carry),
        )

    return rule
