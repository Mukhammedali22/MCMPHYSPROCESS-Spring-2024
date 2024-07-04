## 1D Transport Equation

### The First Scheme Against the Flow Method

This method approximates the 1D transport equation with an error order of \( O(\Delta x, \Delta t) \).

#### Equation

The 1D transport equation is given by:

### `∂u/∂t + c ∂u/∂x = 0, for x ∈ [0, 3], t > 0`

where `∂u/∂t` and `∂u/∂x` represent the partial derivatives of `u` with respect to time `t` and space `x`, respectively.

#### Schemes

##### For `c > 0` (Backward Scheme)

### `(U_i^(n+1) - U_i^n) / Δt = -c (U_i^n - U_{i-1}^n) / Δx`

##### For `c < 0` (Forward Scheme)

### `(U_i^(n+1) - U_i^n) / Δt = -c (U_{i+1}^n - U_i^n) / Δx`

#### Stability Condition

The stability condition for this method is `c Δt / Δx ≤ 1`, where `c` is a constant.

The scheme used depends on the sign of the constant `c`:
- When `c` is positive, a backward scheme is applied.
- When `c` is negative, a forward scheme is used.

The constant `c` represents the velocity at which the solution propagates. In simpler terms, it indicates whether the movement is from left to right (positive `c`) or from right to left (negative `c`).

### GIF Animations

The following GIFs illustrate the behavior of the 1D transport equation under different conditions and schemes. They demonstrate the propagation of the solution over time, highlighting the impact of the constant `c` on the direction and stability of the solution.

![](https://github.com/Mukhammedali22/MCMPHYSPROCESS-Spring-2024/blob/main/Week5/HW5_2_backward.gif)
![](https://github.com/Mukhammedali22/MCMPHYSPROCESS-Spring-2024/blob/main/Week5/HW5_2_forward.gif)
