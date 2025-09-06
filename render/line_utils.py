def projection_lines(feature_lines, tol=1e-4, angle_tol_rel=0.15, min_gap=1e-4):
    """
    CAD2Sketch-style projection lines:
      1) Detect planar quadrilateral faces from existing straight segments.
      2) Pair faces that are (near) parallel.
      3) For each vertex of face A, orthogonally project onto face B along face A's normal.
      4) If the hit lies inside B, create segment [vertex_A, hit_B].
      5) Return only NEW straight lines (skip duplicates/contained/zero-length).

    Args:
        feature_lines : list of primitives (10-value format). Only type==1 used to detect faces.
        tol           : spatial tolerance.
        angle_tol_rel : faces considered parallel if |cross(n1,n2)|/(|n1||n2|) <= angle_tol_rel
                        (≈ sin(max_angle)); e.g., 0.15 ~ 8.6 degrees.
        min_gap       : minimum length for a projection segment (avoid tiny/degenerate).

    Returns:
        List of NEW straight-line primitives: [x1,y1,z1, x2,y2,z2, 0,0,0, 1]
    """

    TYPE_LINE = 1

    # -----------------------
    # basic vector ops
    # -----------------------
    def sub(a,b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
    def add(a,b): return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
    def scl(a,s): return (a[0]*s, a[1]*s, a[2]*s)
    def dot(a,b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
    def cross(a,b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
    def nrm2(a): return dot(a,a)
    def nrm(a):
        d2 = nrm2(a)
        if d2 <= 0.0: return (0.0,0.0,0.0)
        d = d2 ** 0.5
        return (a[0]/d, a[1]/d, a[2]/d)
    def dist2(a,b): return nrm2(sub(a,b))
    def eq_pt(a,b): return dist2(a,b) <= tol*tol

    def eq_seg(a1, a2, b1, b2):
        return (eq_pt(a1,b1) and eq_pt(a2,b2)) or (eq_pt(a1,b2) and eq_pt(a2,b1))

    def point_on_segment(p, a, b):
        ab = sub(b,a); ap = sub(p,a)
        L2 = nrm2(ab)
        if L2 <= tol*tol:
            return eq_pt(p,a)
        t = dot(ap, ab) / L2
        if t < -tol or t > 1+tol:
            return False
        perp = sub(ap, scl(ab, t))
        return nrm2(perp) <= tol*tol

    def seg_contained_in(a1, a2, b1, b2):
        return point_on_segment(a1, b1, b2) and point_on_segment(a2, b1, b2)

    # -----------------------
    # 1) pull straight segments
    # -----------------------
    segs = []
    for v in feature_lines:
        if int(v[9]) == TYPE_LINE:
            p1 = (v[0], v[1], v[2]); p2 = (v[3], v[4], v[5])
            if dist2(p1,p2) > tol*tol:
                segs.append((p1,p2))
    if not segs:
        return []

    # cache existing straight segments for duplicate filtering
    existing_segments = list(segs)

    # -----------------------
    # 2) cluster endpoints into vertices (within tol), build adjacency
    # -----------------------
    vertices = []  # unique points
    def find_or_add(pt):
        for i,q in enumerate(vertices):
            if eq_pt(pt,q): return i
        vertices.append(pt); return len(vertices)-1

    edges = []  # undirected (i,j), i<j
    for (a,b) in segs:
        ia, ib = find_or_add(a), find_or_add(b)
        if ia != ib:
            e = (min(ia,ib), max(ia,ib))
            if e not in edges:
                edges.append(e)

    adj = {i:set() for i in range(len(vertices))}
    for (i,j) in edges:
        adj[i].add(j); adj[j].add(i)

    # -----------------------
    # 3) enumerate simple 4-cycles (a-b-c-d-a)
    # -----------------------
    quads = set()
    for a in range(len(vertices)):
        for b in adj[a]:
            if b == a: continue
            for c in adj[b]:
                if c in (a,b): continue
                for d in adj[c]:
                    if d in (a,b,c): continue
                    if a in adj[d]:
                        cyc = [a,b,c,d]
                        rots = [
                            tuple(cyc),
                            (b,c,d,a),
                            (c,d,a,b),
                            (d,a,b,c),
                            tuple(reversed(cyc)),
                            tuple(reversed((b,c,d,a))),
                            tuple(reversed((c,d,a,b))),
                            tuple(reversed((d,a,b,c))),
                        ]
                        quads.add(min(rots))

    # -----------------------
    # 4) keep planar parallelograms (near), compute face frames
    # -----------------------
    def coplanar(pA,pB,pC,pD):
        n = cross(sub(pB,pA), sub(pC,pA))
        h = dot(n, sub(pD,pA))
        return abs(h) <= tol * (1.0 + (nrm2(n)**0.5))

    def is_parallelogram(pA,pB,pC,pD):
        AB, BC, CD, DA = sub(pB,pA), sub(pC,pB), sub(pD,pC), sub(pA,pD)
        # Opposites parallel: |cross|/(|u||v|) small
        def rel_cross(u,v):
            nu, nv = (nrm2(u)**0.5), (nrm2(v)**0.5)
            if nu <= tol or nv <= tol: return 1.0
            cr = cross(u,v)
            return (nrm2(cr)**0.5) / (nu*nv)
        return (rel_cross(AB,CD) <= angle_tol_rel) and (rel_cross(BC,DA) <= angle_tol_rel)

    faces = []  # each: dict with A,B,C,D, AB,AD, normal n (unit), plane point A
    for (a,b,c,d) in quads:
        A,B,C,D = vertices[a], vertices[b], vertices[c], vertices[d]
        if coplanar(A,B,C,D) and is_parallelogram(A,B,C,D):
            AB, AD = sub(B,A), sub(D,A)
            n = nrm(cross(AB, AD))
            if nrm2(n) <= tol*tol:
                continue
            faces.append({
                "A": A, "B": B, "C": C, "D": D,
                "AB": AB, "AD": AD,
                "n": n
            })
    if not faces:
        return []

    # -----------------------
    # 5) pair parallel faces (|cross(n1,n2)| <= angle_tol_rel)
    # -----------------------
    def normals_parallel(n1, n2):
        cr = cross(n1, n2)
        return (nrm2(cr)**0.5) <= angle_tol_rel  # since n1,n2 are unit-ish

    face_pairs = []
    for i in range(len(faces)):
        for j in range(i+1, len(faces)):
            if normals_parallel(faces[i]["n"], faces[j]["n"]) or normals_parallel(faces[i]["n"], scl(faces[j]["n"], -1.0)):
                face_pairs.append((i, j))
    if not face_pairs:
        return []

    # -----------------------
    # inside test on a parallelogram face via local (u,v) coords:
    # Solve q = A + u*AB + v*AD
    # -----------------------
    def in_face(q, face):
        A, AB, AD = face["A"], face["AB"], face["AD"]
        AQ = sub(q, A)
        # Solve for u,v using least-squares on 2D subspace:
        # We can project onto AB/AD basis by Gram matrix
        a11 = dot(AB, AB); a22 = dot(AD, AD); a12 = dot(AB, AD)
        b1  = dot(AQ, AB); b2  = dot(AQ, AD)
        # Solve [a11 a12; a12 a22] [u v]^T = [b1 b2]^T
        det = a11*a22 - a12*a12
        if abs(det) <= tol:
            return False
        u = ( b1*a22 - b2*a12) / det
        v = (-b1*a12 + b2*a11) / det
        eps = 1e-6
        return (-eps <= u <= 1+eps) and (-eps <= v <= 1+eps)

    # Ray-plane intersection: p + t * dir hits plane (Q0, nQ)
    def project_to_plane(p, dirv, Q0, nQ):
        denom = dot(dirv, nQ)
        if abs(denom) <= tol:  # nearly parallel; skip
            return None
        t = dot(sub(Q0, p), nQ) / denom
        return add(p, scl(dirv, t))

    # -----------------------
    # 6) build projection segments (both directions), filter duplicates
    # -----------------------
    new_lines, new_segments = [], []

    def is_duplicate(p1,p2):
        # too short?
        if dist2(p1,p2) <= max(tol*tol, min_gap*min_gap):
            return True
        # existing
        for (q1,q2) in existing_segments:
            if eq_seg(p1,p2,q1,q2) or seg_contained_in(p1,p2,q1,q2):
                return True
        # newly added
        for (r1,r2) in new_segments:
            if eq_seg(p1,p2,r1,r2) or seg_contained_in(p1,p2,r1,r2):
                return True
        return False

    def make_line(p1, p2):
        return [p1[0],p1[1],p1[2], p2[0],p2[1],p2[2], 0.0,0.0,0.0, TYPE_LINE]

    # vertices of a face
    def face_vertices(face):
        return [face["A"], face["B"], face["C"], face["D"]]

    for (ia, ib) in face_pairs:
        FA, FB = faces[ia], faces[ib]

        # A -> B (along A's normal)
        dirAB = FA["n"]
        for p in face_vertices(FA):
            q = project_to_plane(p, dirAB, FB["A"], FB["n"])
            if q is not None and in_face(q, FB) and not is_duplicate(p, q):
                new_segments.append((p, q))
                new_lines.append(make_line(p, q))

        # B -> A (along B's normal)
        dirBA = FB["n"]
        for p in face_vertices(FB):
            q = project_to_plane(p, dirBA, FA["A"], FA["n"])
            if q is not None and in_face(q, FA) and not is_duplicate(p, q):
                new_segments.append((p, q))
                new_lines.append(make_line(p, q))

    return new_lines
