import numpy as np

class Particle:

    def __init__(self, pos, vel, pbest_pos, pbest_val, informants):

        self.pos = pos                                          # current position
        self.vel = vel                                          # current velocity
        self.pbest_pos = pbest_pos                              # best position seen by this particle
        self.pbest_val = pbest_val                              # best value seen by this particle
        self.informants = informants                            # indices of informants

class PSO:

    # PSO with informants (Algorithm 39)
    def __init__(
        self,
        swarm_size: int,
        dim: int,
        bounds: tuple,                                          # (low, high)
        w: float = 0.729,                                       # inertia
        c1: float = 1.494,                                      # cognitive
        c2: float = 1.494,                                      # social (informants)
        k_informants: int = 3,
        seed: int = 0
    ):
        self.n = swarm_size
        self.dim = dim
        self.bounds = self.norm_bounds(bounds)
        self.w, self.c1, self.c2 = w, c1, c2
        self.k = k_informants
        self.rng = np.random.default_rng(seed)

        self.particles = []
        self.gbest_pos = None
        self.gbest_val = np.inf

    def norm_bounds(self, bounds):

        # turn scalars into arrays
        lo, hi = bounds
        lo = np.full(self.dim, lo) if np.isscalar(lo) else np.asarray(lo, float)
        hi = np.full(self.dim, hi) if np.isscalar(hi) else np.asarray(hi, float)
        return lo, hi

    # random initial position within bounds
    def rand_pos(self):
        lo, hi = self.bounds
        return self.rng.uniform(lo, hi)

    # random initial velocity within +/- half the bounds span
    def rand_vel(self):
        lo, hi = self.bounds
        span = hi - lo
        return self.rng.uniform(-0.5 * span, 0.5 * span)

    def make_informants(self):

        # [A39 L5-6] choose informant sets for each particle. Pick k distinct informants per particle (include self)
        idxs = np.arange(self.n)
        inf = []
        for i in range(self.n):
            picks = self.rng.choice(idxs, size=self.k, replace=False)
            if i not in picks:
                picks[0] = i
            inf.append(np.sort(picks))
        return np.asarray(inf, dtype=int)

    def init_swarm(self, fitness_fn):

        # [A39 L1-4] init particles and personal bests
        inf_mat = self.make_informants()
        self.particles = []
        self.gbest_pos, self.gbest_val = None, np.inf

        for i in range(self.n):
            pos = self.rand_pos()
            vel = self.rand_vel()
            val = float(fitness_fn(pos))
            p = Particle(pos, vel, pos.copy(), val, inf_mat[i])
            self.particles.append(p)
            if val < self.gbest_val:
                self.gbest_val = val
                self.gbest_pos = pos.copy()

    def local_best_of(self, i):

        # [A39 L5-6] best personal best among informants of particle i
        best_val = np.inf
        best_pos = None
        for j in self.particles[i].informants:
            pj = self.particles[j]
            if pj.pbest_val < best_val:
                best_val = pj.pbest_val
                best_pos = pj.pbest_pos
        return best_pos, best_val

    def apply_bounds(self, pos, vel):

        # reflective boundary handling (Not from the A39 pseudocode, this is an addition)
        lo, hi = self.bounds
        p, v = pos.copy(), vel.copy()
        below, above = p < lo, p > hi  
              
        # reflect using matching-index lo/hi
        if np.any(below):
            p_b = p[below]
            lo_b = lo[below]
            p[below] = lo_b + (lo_b - p_b)          # reflect inside
            v[below] *= -1.0                        # flip velocity

        if np.any(above):
            p_a = p[above]
            hi_a = hi[above]
            p[above] = hi_a - (p_a - hi_a)
            v[above] *= -1.0

        # clamp just in case of double reflection
        p = np.minimum(np.maximum(p, lo), hi)
        return p, v

    def step(self, fitness_fn):

        # [A39 L6-14] velocity and position update, then pbest update
        for i, p in enumerate(self.particles):
            lbest_pos, _ = self.local_best_of(i)                    # informants' best

            r1 = self.rng.random(self.dim)                          # fresh randomness
            r2 = self.rng.random(self.dim)

            # velocity update
            cognitive = self.c1 * r1 * (p.pbest_pos - p.pos)
            social    = self.c2 * r2 * (lbest_pos   - p.pos)
            p.vel = self.w * p.vel + cognitive + social             # [A39 L9-10]

            # position update
            p.pos = p.pos + p.vel                                   # [A39 L11]
            p.pos, p.vel = self.apply_bounds(p.pos, p.vel)          # boundary handling

            # evaluate and update personal best
            val = float(fitness_fn(p.pos))
            if val < p.pbest_val:                                   # [A39 L12-13]
                p.pbest_val = val
                p.pbest_pos = p.pos.copy()

            # optional global best track (handy for logging)
            if val < self.gbest_val:                                # [A39 L14]        
                self.gbest_val = val
                self.gbest_pos = p.pos.copy()
