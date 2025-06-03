# ml-accelerated-simulation

Great. I’ll prepare a beginner-friendly walkthrough of the math and numerical methods behind solving the 2D Navier-Stokes equations using the most commonly used approach: the finite difference method. This will cover both the theoretical foundation and practical steps to implement a solver from scratch.

I’ll include an overview of the equations, a step-by-step breakdown of how they are discretized and solved numerically, and a basic code-style implementation plan you can follow or adapt. I’ll let you know when it’s ready.


# Beginner’s Guide: 2D Navier–Stokes Solver with Finite Differences

## Introduction to the 2D Incompressible Navier–Stokes Equations

The **Navier–Stokes equations** describe how fluids move, encapsulating Newton’s second law for fluid motion (momentum conservation) and conservation of mass. In the incompressible 2D case (constant density, no compression), the equations consist of two momentum equations (for velocity components $u$ and $v$ in the $x$ and $y$ directions) and a continuity equation enforcing incompressibility. In continuous form, they can be written as:

* **Momentum (x-direction)**: $\displaystyle \frac{\partial u}{\partial t} = \nu \nabla^2 u - \Big(u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y}\Big);-;\frac{\partial p}{\partial x}$
* **Momentum (y-direction)**: $\displaystyle \frac{\partial v}{\partial t} = \nu \nabla^2 v - \Big(u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y}\Big);-;\frac{\partial p}{\partial y}$
* **Continuity (incompressibility)**: $\displaystyle \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$

Here $u(x,y,t)$ and $v(x,y,t)$ are the fluid’s velocity components, $p(x,y,t)$ is the pressure field, and $\nu$ is the kinematic viscosity (which multiplies the Laplacian term $\nabla^2$, representing diffusion of momentum). Intuitively:

* The **unsteady term** $(\partial/\partial t)$ represents acceleration of a fluid particle.
* The **convective terms** $(u,\partial/\partial x + v,\partial/\partial y)$ represent advection – how the fluid’s own motion transports momentum (this makes the equations nonlinear).
* The **pressure gradient** terms $-\partial p/\partial x$ and $-\partial p/\partial y$ act as forces per unit mass (pressure pushes the fluid).
* The **viscous term** $\nu \nabla^2 u$ (and similarly for $v$) represents diffusion of momentum (viscosity “smooths out” velocity differences).
* The **continuity constraint** $\partial\_x u + \partial\_y v = 0$ means the flow is incompressible (volume of any fluid element remains constant), ensuring no sources or sinks of volume. This constraint couples the $u$ and $v$ fields through pressure: pressure adjusts itself such that the final velocity field has zero divergence (satisfies incompressibility).

Solving these equations **analytically** is generally impossible for most flows, so we turn to numerical methods. We will use the **finite difference method (FDM)** to approximate the equations on a grid and step them forward in time. The goal is to compute $u, v,$ and $p$ over time for a given domain and initial/boundary conditions (for example, the classic *lid-driven cavity* flow, described later). Before diving into the numerical scheme, we need to set up a proper discretization of the domain and variables.

## Discretization: Grid Layout and Time Stepping

**Spatial Discretization:** In finite difference modeling, we overlay a grid on the 2D domain (e.g. a square domain). The continuous fields ($u, v, p$) will be represented by their values at discrete grid points. A simple choice is a uniform Cartesian grid with spacing $\Delta x$ in the $x$-direction and $\Delta y$ in the $y$-direction. An important design decision is whether to use a **collocated** or **staggered** grid for storing variables:

* **Collocated grid:** All variables ($u, v, p$) are stored at the same grid points. This is straightforward, but in incompressible flow it can lead to numerical issues like spurious oscillations in the pressure field and decoupling between pressure and velocity. Special techniques (e.g. Rhie–Chow interpolation or pressure smoothing) are needed to ensure stability on collocated grids.
* **Staggered grid:** Different variables are stored at different locations in each grid cell. A common approach (the Marker-and-Cell scheme) stores pressure at cell centers and velocity components at the cell face midpoints (horizontal velocity $u$ on the vertical cell faces, and vertical velocity $v$ on the horizontal cell faces). This arrangement naturally couples pressure and velocity and avoids the checkerboard pressure decoupling seen in collocated grids. We will use a staggered grid for our solver.

Using a staggered layout means, for example, that a pressure $p\_{i,j}$ is defined at the center of cell $(i,j)$, the $u$-velocity $u\_{i+\frac{1}{2},j}$ is defined at the midpoint of the right face of that cell, and $v\_{i,j+\frac{1}{2}}$ at the midpoint of the top face (with appropriate indexing at boundaries). Figure 1 illustrates a typical staggered grid arrangement (pressure at centers and velocities at face centers). The staggered grid makes it easier to compute derivatives like $\partial p/\partial x$ at the $u$-locations and automatically satisfies discrete mass conservation when the pressure field is solved correctly.

**Time Discretization:** We also divide time into small increments of size $\Delta t$. We will march the solution forward in time, updating from $t^n$ to $t^{n+1}=t^n+\Delta t$. For simplicity, we use an **explicit time-stepping scheme** (forward Euler method), which is first-order accurate in time: the time derivative $\partial u/\partial t$ is approximated by a simple forward difference $\frac{u^{n+1}-u^n}{\Delta t}$. Explicit stepping is easy to implement and conceptually clear, though it requires small $\Delta t$ for stability. (In practice, $\Delta t$ must satisfy a **CFL condition** to keep the simulation stable: the fluid should not advect information more than one grid cell in one time step, and viscous diffusion also imposes a constraint. We won’t delve into detailed stability theory here, but you should choose $\Delta t$ small enough based on the maximum velocity and grid size.)

**Boundary Conditions:** Before we proceed, note that we must specify boundary conditions for $u$, $v$, and $p$ on the domain boundaries. A common choice is **no-slip walls** (fluid sticks to solid boundaries, so $u=v=0$ at walls, except where a lid moves) and a **Neumann condition** for pressure (zero normal derivative, $\partial p/\partial n = 0$, so pressure doesn’t have fake “sources” at the boundary). We will use such conditions in the example. With the grid and time step defined, we can now derive finite-difference equations to approximate the Navier–Stokes equations.

## Finite Difference Approximation of the Navier–Stokes Equations

To solve the equations numerically, we replace the continuous partial derivatives with finite differences on our grid. Consider the $u$-momentum equation (the $x$-component of momentum). In continuous form (from above):

$\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} = -\frac{1}{\rho}\frac{\partial p}{\partial x} + \nu\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right).$

We will derive a discrete version of this equation at a grid point. (For definiteness, assume a staggered grid and derive the equation at a $u$-velocity location, which lies on a cell face between pressure points. The same idea applies to $v$ with roles of $x$ and $y$ swapped.)

**1. Time derivative:** Use a forward difference in time. At time level $n$ (current time) and $n+1$ (next time), the derivative is approximately:

$\frac{\partial u}{\partial t}\Big|_{i,j}^{n} \approx \frac{u_{i,j}^{\,n+1} - u_{i,j}^{\,n}}{\Delta t}.$

This represents the acceleration of the fluid at that point.

**2. Convective (advection) term:** The term $u,\partial\_x u + v,\partial\_y u$ is nonlinear and must be handled carefully on a staggered grid. One simple approach is to use central differences for the spatial derivatives. For example, the $\partial u/\partial x$ term at the $u$-location $(i,j)$ can be approximated by a central difference using neighboring $u$ values (to the left and right along the $x$ direction). If $u\_{i,j}$ is at a face, its immediate neighbors in the $x$ direction are $u\_{i,j}$ itself and $u\_{i+1,j}$ at adjacent faces; however, since $u$ is staggered, these neighbors align with pressure points. We can interpolate velocities to the same location before differencing. A common technique is to compute convective fluxes by averaging adjacent values onto a common grid. For instance, one can average $u$ values to cell centers or vice versa so that products like $u \cdot (\partial u/\partial x)$ use collocated quantities.

For simplicity, assume we have collocated values to illustrate the finite difference. A central difference for the derivative is:

$\frac{\partial u}{\partial x}\Big|_{i,j} \approx \frac{u_{i+1,j}^{\,n} - u_{i-1,j}^{\,n}}{2\,\Delta x}.$

Likewise, $\partial u/\partial y$ at $(i,j)$ can be approximated by $\frac{u\_{i,j+1}^{,n} - u\_{i,j-1}^{,n}}{2,\Delta y}$. The convective term $u,\partial\_x u$ at $(i,j)$ can then be written as $u\_{i,j}^n \cdot \frac{u\_{i+1,j}^n - u\_{i-1,j}^n}{2\Delta x}$, and similarly $v,\partial\_y u \approx v\_{i,j}^n \cdot \frac{u\_{i,j+1}^n - u\_{i,j-1}^n}{2\Delta y}$. (On a staggered grid, the $v$ value needed here is located at a half-cell shift; one would average nearby $v$ values to estimate $v$ at the $u$-location. These interpolations are a detail of staggered grids, but the core idea is we use differences of neighboring values to approximate derivatives.)

If stability is a concern, one may use **upwind differencing** for the convective terms (using one-sided differences based on the flow direction) to avoid oscillations. However, upwind schemes introduce numerical diffusion and lower accuracy. As a first implementation, many 2D tutorials use central differences and small time steps (assuming low Reynolds number or adding a bit of artificial viscosity if needed).

**3. Pressure gradient term:** On the staggered grid, the pressure gradient $\partial p/\partial x$ at a $u$-location can be found by difference between the pressures in the neighboring cell centers. If $p\_{i,j}$ and $p\_{i+1,j}$ are the pressures in the cells to the left and right of the $u$-face, then:

$\frac{\partial p}{\partial x}\Big|_{i,j} \approx \frac{p_{i+1,j}^{\,n} - p_{i,j}^{\,n}}{\Delta x},$

evaluated at the half-step between the two pressure points (which is exactly where $u\_{i,j}$ lives). Similarly $\partial p/\partial y$ for the $v$-equation would use pressure difference in the vertical direction. This ensures pressure forces drive the velocity change. (Using the staggered arrangement means we don’t need to interpolate pressure for this difference – it’s naturally centered on the $u$ location by the geometry.)

**4. Viscous diffusion term:** The second derivative terms $\partial^2 u/\partial x^2$ and $\partial^2 u/\partial y^2$ are approximated by **central second differences**. For example, the second derivative in $x$ at $(i,j)$ can be written as:

$\frac{\partial^2 u}{\partial x^2}\Big|_{i,j} \approx \frac{u_{i+1,j}^{\,n} - 2\,u_{i,j}^{\,n} + u_{i-1,j}^{\,n}}{\Delta x^2},$

with an analogous formula in $y$. These formulas come from the standard second-order finite difference approximation of the Laplacian on a grid. Using central differences ensures second-order spatial accuracy for the diffusion term.

Now we combine all these pieces into the **discrete momentum equation for $u$** at point $(i,j)$:

$$
\begin{aligned}
\frac{u_{i,j}^{\,n+1} - u_{i,j}^{\,n}}{\Delta t} &= -\,\Big(u_{i,j}^n \frac{u_{i+1,j}^n - u_{i-1,j}^n}{2\Delta x} + v_{i,j}^n \frac{u_{i,j+1}^n - u_{i,j-1}^n}{2\Delta y}\Big)\;-\;\frac{1}{\rho}\frac{p_{i+1,j}^n - p_{i,j}^n}{\Delta x} \\
&\quad +\; \nu \Big(\frac{u_{i+1,j}^n - 2u_{i,j}^n + u_{i-1,j}^n}{\Delta x^2} + \frac{u_{i,j+1}^n - 2u_{i,j}^n + u_{i,j-1}^n}{\Delta y^2}\Big)\,. 
\end{aligned} 
$$

This is a single algebraic equation that can be rearranged to solve for $u\_{i,j}^{,n+1}$ (since all terms on the right are known from time level $n$). A similar equation can be written for the $v$-velocity at its grid position. These are the **discretized momentum equations** (sometimes called the Navier–Stokes update formulas). They tell us how to compute the tentative new velocities from the old velocities and pressure.

However, there’s a catch: if we update $u$ and $v$ using these equations directly, the new velocities $u^{n+1}, v^{n+1}$ will not automatically satisfy the continuity equation $\nabla\cdot \mathbf{u}=0$. In other words, after this provisional update, the flow may become compressible (non-zero divergence) due to the way we treated pressure. We need to enforce incompressibility at each time step. This is where the **pressure Poisson equation and projection method** come in.

## Enforcing Incompressibility: Pressure Poisson Equation & Projection Method

The **projection method** is a strategy to enforce the divergence-free condition (continuity) by using pressure as a correction tool. The key idea is to split the update into two steps:

1. **Predict** an intermediate velocity field $(u^*, v^*)$ by ignoring (or guessing) the pressure influence. This usually means solving the momentum equations without the pressure term (or using the old pressure), yielding a velocity that **does not necessarily satisfy** $\nabla \cdot \mathbf{u}^{*} = 0$.
2. **Correct** the intermediate velocity by computing a pressure field $p^{n+1}$ that will enforce incompressibility. The pressure adjustment is found by solving a Poisson equation, and then the velocity is adjusted (projected) to be divergence-free: $\mathbf{u}^{n+1} = \mathbf{u}^{*} - \frac{\Delta t}{\rho}\nabla p^{n+1}$. This projection step ensures that $\nabla \cdot \mathbf{u}^{n+1} = 0$.

In our finite difference scheme, we effectively already derived the “predictor” step by excluding the new-time pressure in the momentum update. To make it concrete, we can implement the method as follows:

* **Intermediate velocity calculation:** Compute $u^*$ and $v^*$ using the momentum equations **without** the pressure term. That is, use the convective and viscous terms to advance the velocity:

  $u^*_{i,j} = u^n_{i,j} + \Delta t \Big[-(u\partial_x u + v\partial_y u)^n_{i,j} + \nu(\partial^2_x u + \partial^2_y u)^n_{i,j}\Big]$

  (and similarly for $v^*$). Now $u^*$ and $v^*$ are a provisional velocity field at the new time that likely does not satisfy $\partial\_x u^{*} + \partial\_y v^{*} = 0$. This is essentially what step (1) in our algorithm will do.

* **Pressure Poisson equation:** To enforce incompressibility, we solve for a new pressure $p^{n+1}$ such that if we adjust $u^*$ and $v^*$ by the pressure gradients, the result is divergence-free. Taking the divergence of the velocity correction formula $\mathbf{u}^{n+1} = \mathbf{u}^{*} - \frac{\Delta t}{\rho}\nabla p^{n+1}$ and requiring $\nabla \cdot \mathbf{u}^{n+1} = 0$ gives:

  $\nabla \cdot \mathbf{u}^{n+1} = \nabla \cdot \mathbf{u}^* - \frac{\Delta t}{\rho}\,\nabla \cdot (\nabla p^{n+1}) = 0.$

  But $\nabla \cdot (\nabla p) = \nabla^2 p$ (the Laplacian of $p$). So we obtain a **Poisson equation for pressure**:

  $\nabla^2 p^{\,n+1} = \frac{\rho}{\Delta t}\,\nabla \cdot \mathbf{u}^*,$

  which we must solve for $p^{n+1}$ in the domain. In words: the pressure field is determined by the requirement that it will remove any divergence in the intermediate velocity. The right-hand side $\nabla \cdot \mathbf{u}^{*}/(\Delta t)$ is computed from the intermediate velocities (we know these values), and $\nabla^2 p$ is expanded in finite differences (a second derivative in $x$ plus second derivative in $y$ of $p$). This is an elliptic equation for $p^{n+1}$ that we will solve at each time step.

  **Discrete form of the Poisson equation:** On our grid, the Laplacian of pressure can be approximated by central differences, similar to the diffusion term earlier. For interior points $(i,j)$, the discrete Poisson equation becomes:

  $\frac{p_{i+1,j}^{\,n+1} - 2p_{i,j}^{\,n+1} + p_{i-1,j}^{\,n+1}}{\Delta x^2} + \frac{p_{i,j+1}^{\,n+1} - 2p_{i,j}^{\,n+1} + p_{i,j-1}^{\,n+1}}{\Delta y^2} = \frac{\rho}{\Delta t}\Big(\frac{u^*_{i,j} - u^*_{i-1,j}}{\Delta x} + \frac{v^*_{i,j} - v^*_{i,j-1}}{\Delta y}\Big),$

  where we used a finite difference approximation for the divergence $\partial\_x u^{*} + \partial\_y v^{*}$ at the cell center (note: $u^**{i,j}$ on the right is the $x$-velocity through the right face of cell $(i-1,j)$, and $u^**{i-1,j}$ is through the left face; their difference over $\Delta x$ gives a discrete approximation to $\partial\_x u{*}$ at the cell center, and similarly for the $v^*$ term in $y$). This large linear system can be solved by iterative solvers such as Gauss–Seidel or Jacobi relaxation, since direct solution can be slow for fine grids. We assume appropriate pressure boundary conditions (e.g. $\partial p/\partial n = 0$ at walls, or setting a reference pressure point to zero to fix the constant).

* **Velocity correction (projection):** Once $p^{n+1}$ is obtained, we correct the intermediate velocities to enforce incompressibility:

  $u^{n+1}_{i,j} = u^*_{i,j} - \frac{\Delta t}{\rho}\,\frac{p^{\,n+1}_{i+1,j} - p^{\,n+1}_{i,j}}{\Delta x},$

  $v^{n+1}_{i,j} = v^*_{i,j} - \frac{\Delta t}{\rho}\,\frac{p^{\,n+1}_{i,j+1} - p^{\,n+1}_{i,j}}{\Delta y},$

  using finite differences for the pressure gradients at the $u$ and $v$ locations (similar to how we computed them in the momentum equation). After this step, by construction, the continuity equation at time $n+1$ is satisfied (within numerical tolerance). We have effectively “projected” $\mathbf{u}^{*}$ onto a divergence-free space using the pressure gradient. This completes the time step.

The beauty of the projection method is that it **decouples velocity and pressure computations**: we first solve for velocities ignoring pressure, then solve a scalar Poisson equation for pressure, then correct velocities. The trade-off is that we have to solve an additional Poisson system each time step, but it allows using efficient solvers and avoids directly inverting a large coupled system. At the end of each time step, we have new $(u^{n+1}, v^{n+1}, p^{n+1})$ that satisfy both momentum and incompressibility.

## Solver Algorithm (Pseudocode)

Putting everything together, here is a step-by-step outline for a simple 2D Navier–Stokes solver using finite differences and the projection method:

1. **Initialization:** Set up the grid with appropriate resolution (spacing $\Delta x$, $\Delta y$) and initialize the velocity and pressure fields. Apply initial conditions (e.g. initially $u=v=0$ everywhere, or some predefined flow). Set the fluid properties (density $\rho$, viscosity $\nu$), choose a time step $\Delta t$, and specify boundary conditions for $u$, $v$, and $p$ (no-slip walls, lid velocity, etc.). Compute any derived parameters or helper arrays (none are strictly needed for this basic solver aside from maybe an array for the Poisson RHS).

2. **Time-stepping loop:** For each time step $n \to n+1$:
   a. **Compute intermediate velocities ($u^*, v^*$):** Using the velocity field at time $n$, calculate the convective terms and diffusive terms at each interior grid point. Then update the velocities explicitly:
   $u^* = u^n + \Delta t\Big[-(u \partial_x u + v \partial_y u)^n + \nu(\partial^2_x u + \partial^2_y u)^n\Big],$
   and similarly for $v^*$. This yields $u^*(x,y)$ and $v^*(x,y)$ at the new time, but they are not yet divergence-free.
   b. **Solve pressure Poisson equation:** Form the source term for pressure from the intermediate velocity field: $b\_{i,j} = \frac{\rho}{\Delta t}\Big(\frac{u^**{i,j} - u^**{i-1,j}}{\Delta x} + \frac{v^*\_{i,j} - v^**{i,j-1}}{\Delta y}\Big)$ for each cell center $(i,j)$. This $b$ is the discrete divergence of $u^{*}$ (multiplied by $\rho/\Delta t$). Solve the linear system:
   $\nabla^2 p^{\,n+1} = b,$
   for the new pressure $p^{n+1}$ inside the domain (with $\partial p/\partial n = 0$ at boundaries, or other appropriate pressure BCs). This step typically requires an iterative solver. You iterate the pressure solution until convergence (the residual of the Poisson equation is below a tolerance).
   c. **Pressure projection (velocity correction):** Using $p^{n+1}$, update the intermediate velocities to enforce incompressibility:
   $u^{n+1}_{i,j} = u^*_{i,j} - \frac{\Delta t}{\rho}\frac{p^{\,n+1}_{i+1,j} - p^{\,n+1}_{i,j}}{\Delta x},$
   $v^{n+1}_{i,j} = v^*_{i,j} - \frac{\Delta t}{\rho}\frac{p^{\,n+1}_{i,j+1} - p^{\,n+1}_{i,j}}{\Delta y}.$
   This subtracts the pressure gradient, causing the velocity field to adjust and satisfy $\partial\_x u^{n+1} + \partial\_y v^{n+1}=0$.
   d. **Apply boundary conditions:** Enforce the boundary conditions on $u^{n+1}, v^{n+1}$ (for example, set $u=v=0$ on solid walls, or $u=U*{\text{lid}}$ on a moving lid, etc.). Also handle pressure boundary conditions (often zero Neumann at walls).
   e. **(Optional) Check convergence or record data:** If you are running to a steady state, you might check if the velocity changes are below a threshold and break the loop. Otherwise, you can record the fields or derived quantities (like drag, kinetic energy, etc.) for analysis.

3. **Post-processing:** After the time-stepping loop, you have the velocity and pressure fields over time (or at final time). You can visualize the velocity vectors, streamlines, pressure contours, etc., to interpret the results or verify against known solutions.

This procedure is a simple **projection method algorithm** for the incompressible Navier–Stokes equations. Each cycle advances the solution by $\Delta t$. The most computationally expensive part is solving the Poisson equation for pressure at every time step, but for moderate grid sizes (e.g. $50\times 50$ or $100\times 100$) an iterative solver is manageable.

**Note on implementation:** It’s important to ensure your indexing aligns with the chosen grid layout (staggered vs collocated). With staggered grids, computing the divergence and pressure gradients requires careful handling at the boundaries and interpolation of velocities at cell centers for the source term. The pseudocode above glosses over some index details for clarity. If implementing from scratch, consider starting with a collocated grid for simplicity, using a known trick to stabilize pressure (like adding a small amount of implicit diffusion to pressure or using a specialized interpolation) – or stick with staggered and be diligent with how you loop over interior points for each variable.

## Example: Lid-Driven Cavity Flow

One of the most common test cases for a 2D incompressible flow solver is the **lid-driven cavity flow**. Imagine a square cavity (a box) filled with fluid. The top boundary (lid) moves steadily to the right with a fixed velocity, while all other walls are stationary (no-slip). This motion drives a recirculating flow inside the cavity: the fluid adheres to the moving lid, gets dragged along, then circulates down and back around in a swirling pattern. A primary vortex typically forms in the center of the cavity, and depending on the Reynolds number (ratio of inertial to viscous forces), smaller secondary vortices can appear in the corners. The pressure is higher near the upper left corner (where the lid collides with the side wall, stagnating the flow) and lower at the upper right corner (where the fast-moving fluid tries to pull away), establishing a pressure gradient that balances the lid’s dragging action.

Using the finite difference solver outlined above, we can simulate this scenario. We set $U\_{\text{lid}}$ (top lid velocity), no-slip ($u=v=0$) on the other three walls, and start with fluid at rest. As time evolves, the flow develops and eventually reaches a steady state where the pattern doesn’t change further.

&#x20;*Streamfunction contours for a steady-state lid-driven cavity flow at Reynolds number $200$, computed on a $41\times 41$ grid (staggered layout). The lid (top boundary) moves to the right, driving a circulating flow inside the square cavity. The contour lines (dashed) represent the streamfunction – essentially flow streamlines – showing a large central clockwise vortex. Weaker eddies form in the lower corners at higher Reynolds numbers. This classic result validates the solver’s ability to handle wall boundary conditions and enforce incompressibility.*

To interpret the image: the circular contours indicate a single large recirculation cell. The fluid sticks to the boundaries (zero velocity at walls except the moving lid). Near the lid, flow is pulled rightward; it then turns downward along the right wall, slows down near the bottom, and comes back left along the bottom wall, finally rising along the left wall to complete the loop. A peak velocity occurs near the lid’s midpoint, and a low-velocity region is found in the corners. The pressure (not shown) would be highest near the upper left and lowest near upper right, driving a pressure-driven return flow under the lid. Quantitatively, one can compare the centerline velocity profiles or vortex strength against benchmark data for the lid-driven cavity to ensure the solver is producing accurate results – this is a common validation for CFD codes.

Through this example, we see how the finite difference Navier–Stokes solver captures the essential physics: the conversion of the lid’s motion into a circulating flow and the establishment of a pressure field to enforce incompressibility. By adjusting parameters (grid resolution, time step, Reynolds number), one can explore different regimes – for instance, at higher Reynolds numbers the flow might develop oscillations or additional vortices. The step-by-step approach presented here forms a foundation that can be expanded (to 3D, to more complex boundary shapes, or to using more advanced numerical schemes), but it covers the core concepts: discretization, finite difference approximations, pressure-velocity coupling, and the algorithmic structure of a CFD solver for incompressible flows.

## References

* Harlow, F. & Welch, J. (1965). *Numerical calculation of time-dependent viscous incompressible flow of fluid with free surface.* Phys. Fluids, 8(12). (Introduced the staggered grid “MAC” method for incompressible flow.)
* Chorin, A. (1967). *The numerical solution of the Navier–Stokes equations for an incompressible fluid.* Bull. Amer. Math. Soc., 73(6): 928–931. (Introduced the projection method.)
* Barba, L. A. (2014). *CFD Python: 12 steps to Navier-Stokes.* (Educational notebooks illustrating a simple cavity flow solver using finite differences).
* MIT OpenCourseWare 2.29 (2015). *Lecture 17: Pressure-Velocity Coupling.* (Discussion of collocated vs staggered grids in CFD).
* Columbia University Mechanical Engineering (2016). *Finite Difference Solution of 2D Navier-Stokes (Lei).* (Project report using projection method on staggered grid).
