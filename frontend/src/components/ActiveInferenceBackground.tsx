import { useEffect, useRef } from 'react';

/*
 * Ambient belief-network background.
 *
 * Sparse nodes + thin edges evoke POMDP latent-state transitions.
 * Motion is slow and purposeful — the system is continuously inferring,
 * not performing.
 *
 * Uses window-level mousemove so interaction works behind UI panels.
 * Canvas is fixed + pointer-events-none so it never blocks input.
 */

interface Node {
  bx: number;
  by: number;
  x: number;
  y: number;
  rx: number;
  ry: number;
  speed: number;
  phase: number;
  belief: number;
  beliefTarget: number;
  pulsePhase: number;
  pulseSpeed: number;
  radius: number;
}

interface Edge {
  a: number;
  b: number;
  flow: number;
  flowTarget: number;
  nextChange: number;
}

const NODE_COUNT = 42;
const EDGE_DENSITY = 0.17;
const MOUSE_RADIUS = 200;
const MOUSE_STRENGTH = 26;

export function ActiveInferenceBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mouseRef = useRef({ x: -9999, y: -9999 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;

    let w = 0, h = 0, dpr = 1, animId = 0;
    let nodes: Node[] = [];
    let edges: Edge[] = [];

    function init() {
      dpr = window.devicePixelRatio || 1;
      w = canvas!.clientWidth;
      h = canvas!.clientHeight;
      canvas!.width = w * dpr;
      canvas!.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      nodes = [];
      for (let i = 0; i < NODE_COUNT; i++) {
        nodes.push({
          bx: w * 0.04 + Math.random() * w * 0.92,
          by: h * 0.04 + Math.random() * h * 0.92,
          x: 0, y: 0,
          rx: 28 + Math.random() * 58,
          ry: 18 + Math.random() * 42,
          speed: 0.000045 + Math.random() * 0.00009,
          phase: Math.random() * Math.PI * 2,
          belief: 0.2 + Math.random() * 0.8,
          beliefTarget: 0.2 + Math.random() * 0.8,
          pulsePhase: Math.random() * Math.PI * 2,
          pulseSpeed: 0.00035 + Math.random() * 0.00055,
          radius: 1.0 + Math.random() * 1.6,
        });
      }

      edges = [];
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          if (Math.random() < EDGE_DENSITY) {
            edges.push({
              a: i, b: j,
              flow: Math.random() * 0.3,
              flowTarget: Math.random() > 0.45 ? 0.1 + Math.random() * 0.45 : 0,
              nextChange: performance.now() + 4000 + Math.random() * 11000,
            });
          }
        }
      }
    }

    function onMouseMove(e: MouseEvent) {
      mouseRef.current = { x: e.clientX, y: e.clientY };
    }

    function onMouseLeave() {
      mouseRef.current = { x: -9999, y: -9999 };
    }

    function draw(now: number) {
      ctx.clearRect(0, 0, w, h);
      const { x: mx, y: my } = mouseRef.current;

      for (const n of nodes) {
        n.x = n.bx + Math.sin(now * n.speed + n.phase) * n.rx;
        n.y = n.by + Math.cos(now * n.speed * 0.72 + n.phase + 1.3) * n.ry;

        const dx = n.x - mx;
        const dy = n.y - my;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < MOUSE_RADIUS && dist > 0) {
          const force = (1 - dist / MOUSE_RADIUS) * MOUSE_STRENGTH;
          n.x += (dx / dist) * force;
          n.y += (dy / dist) * force;
        }

        n.belief += (n.beliefTarget - n.belief) * 0.0014;
        if (Math.abs(n.belief - n.beliefTarget) < 0.02) {
          n.beliefTarget = 0.2 + Math.random() * 0.8;
        }
        n.pulsePhase += n.pulseSpeed;
      }

      // Edges — barely visible lines suggesting information flow
      for (const e of edges) {
        e.flow += (e.flowTarget - e.flow) * 0.0018;
        if (now > e.nextChange) {
          e.flowTarget = Math.random() > 0.42 ? 0.08 + Math.random() * 0.42 : 0;
          e.nextChange = now + 5000 + Math.random() * 13000;
        }
        if (e.flow < 0.012) continue;

        const na = nodes[e.a], nb = nodes[e.b];
        const alpha = e.flow * ((na.belief + nb.belief) / 2) * 0.14;

        ctx.beginPath();
        ctx.moveTo(na.x, na.y);
        ctx.lineTo(nb.x, nb.y);
        ctx.strokeStyle = `rgba(74, 124, 111, ${alpha})`;
        ctx.lineWidth = 0.4;
        ctx.stroke();
      }

      // Nodes — tiny pulsing dots varying with belief strength
      for (const n of nodes) {
        const pulse = 1 + Math.sin(n.pulsePhase) * 0.22 * n.belief;
        const r = n.radius * pulse;
        const alpha = 0.09 + n.belief * 0.2;

        // Soft glow on high-belief nodes
        if (n.belief > 0.62) {
          const grad = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, r * 7);
          grad.addColorStop(0, `rgba(74, 124, 111, ${n.belief * 0.022})`);
          grad.addColorStop(1, 'rgba(74, 124, 111, 0)');
          ctx.beginPath();
          ctx.arc(n.x, n.y, r * 7, 0, Math.PI * 2);
          ctx.fillStyle = grad;
          ctx.fill();
        }

        ctx.beginPath();
        ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(74, 124, 111, ${alpha})`;
        ctx.fill();
      }

      animId = requestAnimationFrame(draw);
    }

    init();
    window.addEventListener('resize', init);
    window.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseleave', onMouseLeave);
    animId = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', init);
      window.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseleave', onMouseLeave);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 w-full h-full pointer-events-none"
      style={{ zIndex: 0 }}
    />
  );
}
