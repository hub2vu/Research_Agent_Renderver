/**
 * GraphCanvas Component
 *
 * D3.js force-directed graph visualization.
 */

import React, { useEffect, useRef, useCallback, useState } from 'react';
import * as d3 from 'd3';
import { GraphNode, GraphEdge } from '../lib/mcp';

interface ClusterCenters {
  [clusterId: string]: { x: number; y: number };
}

interface GraphCanvasProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  mode: 'global' | 'paper';
  centerId?: string;
  selectedNodeId?: string;

  // nodeKey(stableKey 우선) -> hex color
  nodeColorMap?: Record<string, string>;

  // Cluster centers for cluster force (NeurIPS mode)
  clusterCenters?: ClusterCenters;

  onNodeClick?: (node: GraphNode) => void;
  onNodeDoubleClick?: (node: GraphNode) => void;
}

/* -------------------------- Constants --------------------------- */

const CLUSTER_COLORS = [
  '#4299e1', '#48bb78', '#ed8936', '#9f7aea',
  '#f56565', '#38b2ac', '#ed64a6', '#667eea'
];

const clusterColorIndex = (d: any): number => {
  const n = CLUSTER_COLORS.length;
  const c = Number((d as any).cluster ?? 0);
  return ((c % n) + n) % n;
};

const isHexColor = (v: unknown): v is string =>
  typeof v === 'string' && /^#[0-9a-fA-F]{6}$/.test(v);

export default function GraphCanvas({
  nodes,
  edges,
  mode,
  centerId,
  selectedNodeId,
  nodeColorMap,
  clusterCenters,
  onNodeClick,
  onNodeDoubleClick
}: GraphCanvasProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const simulationRef = useRef<d3.Simulation<any, any> | null>(null);

  // zoom behavior + last transform
  const zoomBehaviorRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const zoomSelectionRef = useRef<d3.Selection<SVGSVGElement, unknown, null, undefined> | null>(null);
  const zoomTransformRef = useRef<d3.ZoomTransform>(d3.zoomIdentity);

  // node selection (for highlight-only updates)
  const nodeSelectionRef = useRef<d3.Selection<SVGGElement, any, SVGGElement, unknown> | null>(null);

  // cache node positions across rebuilds (keyed by stableKey first)
  const posCacheRef = useRef<Map<string, { x?: number; y?: number; vx?: number; vy?: number }>>(
    new Map()
  );

  // refs for stable resolvers (so nodeColorMap change doesn't recreate main effect)
  const nodeColorMapRef = useRef<Record<string, string>>(nodeColorMap ?? {});
  const centerIdRef = useRef<string | undefined>(centerId);

  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const { width, height } = dimensions;

  /* --------------------- keep refs in sync --------------------- */

  useEffect(() => {
    nodeColorMapRef.current = nodeColorMap ?? {};
  }, [nodeColorMap]);

  useEffect(() => {
    centerIdRef.current = centerId;
  }, [centerId]);

  /* --------------------- Responsive sizing --------------------- */

  useEffect(() => {
    const updateDimensions = () => {
      if (!containerRef.current) return;
      const { clientWidth, clientHeight } = containerRef.current;
      setDimensions({
        width: clientWidth || 800,
        height: clientHeight || 600
      });
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  /* ------------------------ Force Params ------------------------ */

  const getForceParams = useCallback(() => {
    if (mode === 'global') {
      return { linkDistance: 60, chargeStrength: -300, centerStrength: 0.05 };
    }
    return { linkDistance: 80, chargeStrength: -500, centerStrength: 0.08 };
  }, [mode]);

  /* --------------------- Helpers (stable) --------------------- */

  // stable key (for color/pos cache): prefer stableKey -> fallback id
  const nodeKey = useCallback((d: any): string => {
    return String((d as any).stableKey ?? d?.id ?? '');
  }, []);

  const isCenterNode = useCallback((d: any) => {
    const cid = centerIdRef.current;
    return Boolean((d as any).is_center || (d as any).isCenter) || (cid ? d?.id === cid : false);
  }, []);

  const getNodeFill = useCallback((d: any) => {
    // center stays fixed red (원하면 override 허용하도록 바꿀 수 있음)
    if (isCenterNode(d)) return '#f56565';

    const key = nodeKey(d);

    // ✅ stableKey 기반 매핑 우선
    const map = nodeColorMapRef.current || {};
    const byKey = map[key];
    if (isHexColor(byKey)) return byKey;

    // ✅ backward-compat: 예전에는 id로 저장했을 수도 있으니 fallback
    const byId = map[d?.id];
    if (isHexColor(byId)) return byId;

    return CLUSTER_COLORS[clusterColorIndex(d)];
  }, [isCenterNode, nodeKey]);

  const cachePositions = useCallback(() => {
    const sim = simulationRef.current;
    if (!sim) return;

    const m = new Map<string, { x?: number; y?: number; vx?: number; vy?: number }>();
    (sim.nodes() as any[]).forEach((n) => {
      const key = nodeKey(n);
      if (!key) return;
      m.set(key, { x: n.x, y: n.y, vx: n.vx, vy: n.vy });
    });
    posCacheRef.current = m;
  }, [nodeKey]);

  const updateNodeStyles = useCallback(() => {
    const sel = nodeSelectionRef.current;
    if (!sel) return;

    sel.select('circle')
      .attr('r', (d: any) => {
        if (isCenterNode(d)) return 20;
        if (d.id === selectedNodeId) return 18;
        return 12;
      })
      .attr('fill', (d: any) => getNodeFill(d))
      .attr('stroke', (d: any) => (d.id === selectedNodeId ? '#000' : '#fff'))
      .attr('stroke-width', 2);
  }, [selectedNodeId, isCenterNode, getNodeFill]);

  /* -------------------------- Main D3 --------------------------- */
  // ✅ Rebuild only when topology/layout changes (NOT on select/color changes)
  useEffect(() => {
    if (!svgRef.current) return;

    // cache current positions before rebuild
    cachePositions();

    const svg = d3.select(svgRef.current);

    // init zoom once
    if (!zoomBehaviorRef.current) {
      const zoom = d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.1, 4])
        .on('zoom', (event) => {
          zoomTransformRef.current = event.transform;
          svg.select('g.canvas-root').attr('transform', event.transform.toString());
        });

      zoomBehaviorRef.current = zoom;
      zoomSelectionRef.current = svg;
      svg.call(zoom);
    }

    // clear old graph but keep zoom behavior
    svg.selectAll('g.canvas-root').remove();

    // new root, restore last camera transform
    const container = svg.append('g')
      .attr('class', 'canvas-root')
      .attr('transform', zoomTransformRef.current.toString());

    // restore positions by stableKey first (fallback id)
    const graphNodes = nodes.map((n) => {
      const key = nodeKey(n);
      const cached = posCacheRef.current.get(key) ?? posCacheRef.current.get((n as any).id);
      return cached ? ({ ...n, ...cached } as any) : ({ ...n } as any);
    });

    const graphEdges = edges.map((e) => ({ ...e }));

    const { linkDistance, chargeStrength, centerStrength } = getForceParams();

    const simulation = d3.forceSimulation(graphNodes as any)
      .force('link', d3.forceLink(graphEdges as any).id((d: any) => d.id).distance(linkDistance))
      .force('charge', d3.forceManyBody().strength(chargeStrength))
      .force(
        'center',
        (d3.forceCenter(width / 2, height / 2) as any).strength?.(centerStrength) ??
          d3.forceCenter(width / 2, height / 2)
      )
      .force('collide', d3.forceCollide().radius(30));

    // Add cluster force if clusterCenters provided
    if (clusterCenters && Object.keys(clusterCenters).length > 0) {
      const clusterStrength = 0.15;

      simulation
        .force('clusterX', d3.forceX<any>((d) => {
          const clusterId = String(d.cluster ?? 0);
          const center = clusterCenters[clusterId];
          return center ? center.x : width / 2;
        }).strength(clusterStrength))
        .force('clusterY', d3.forceY<any>((d) => {
          const clusterId = String(d.cluster ?? 0);
          const center = clusterCenters[clusterId];
          return center ? center.y : height / 2;
        }).strength(clusterStrength));
    }

    simulationRef.current = simulation;

    // pin center in paper mode
    if (mode === 'paper' && centerId) {
      const centerNode = graphNodes.find((nn: any) => nn.id === centerId);
      if (centerNode) {
        centerNode.fx = width / 2;
        centerNode.fy = height / 2;
      }
    }

    // edges
    const link = container.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(graphEdges)
      .enter()
      .append('line')
      .attr('stroke', '#cbd5e0')
      .attr('stroke-opacity', 0.6)
      .attr('stroke-width', (d: any) => (d.type === 'references' ? 1 : 2));

    // nodes
    const node = container.append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(graphNodes)
      .enter()
      .append('g')
      .style('cursor', 'pointer')
      .call(
        d3.drag<any, any>()
          .on('start', (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on('drag', (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on('end', (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            if (mode !== 'paper' || d.id !== centerId) {
              d.fx = null;
              d.fy = null;
            }
          })
      );

    nodeSelectionRef.current = node as any;

    node.append('circle')
      .attr('r', (d: any) => {
        if (isCenterNode(d)) return 20;
        if (d.id === selectedNodeId) return 18;
        return 12;
      })
      .attr('fill', (d: any) => getNodeFill(d))
      .attr('stroke', (d: any) => (d.id === selectedNodeId ? '#000' : '#fff'))
      .attr('stroke-width', 2);

    node.append('text')
      .attr('dy', 25)
      .attr('text-anchor', 'middle')
      .attr('font-size', '10px')
      .attr('fill', '#4a5568')
      .text((d: any) => {
        const title = d.title || d.id;
        return title.length > 20 ? title.slice(0, 20) + '...' : title;
      });

    node.on('click', (event, d: any) => {
      event.stopPropagation();
      onNodeClick?.(d as GraphNode);
    });

    node.on('dblclick', (event, d: any) => {
      event.stopPropagation();
      onNodeDoubleClick?.(d as GraphNode);
    });

    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => (d.source as any).x)
        .attr('y1', (d: any) => (d.source as any).y)
        .attr('x2', (d: any) => (d.target as any).x)
        .attr('y2', (d: any) => (d.target as any).y);

      node.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
    });

    return () => {
      simulation.stop();
    };
  }, [
    nodes,
    edges,
    mode,
    centerId,
    width,
    height,
    getForceParams,
    onNodeClick,
    onNodeDoubleClick,
    cachePositions,
    isCenterNode,
    getNodeFill,
    nodeKey,
    clusterCenters
    // ❌ selectedNodeId/nodeColorMap 제외: 튐 방지 핵심
  ]);

  /* ------------------- Style-only updates ------------------- */

  // selected 변경 → 스타일만
  useEffect(() => {
    updateNodeStyles();
  }, [updateNodeStyles]);

  // 색 변경 → 스타일만 (simulation 재시작 X)
  useEffect(() => {
    updateNodeStyles();
  }, [nodeColorMap, updateNodeStyles]);

  /* ---------------- Auto-Zoom to Center (paper) ---------------- */
  useEffect(() => {
    if (
      !centerId ||
      mode !== 'paper' ||
      !simulationRef.current ||
      !zoomBehaviorRef.current ||
      !zoomSelectionRef.current
    ) {
      return;
    }

    const sim = simulationRef.current;
    const svg = zoomSelectionRef.current;
    const zoom = zoomBehaviorRef.current;

    const centerNode: any = (sim.nodes() as any[]).find((n) => n.id === centerId);
    if (!centerNode || centerNode.x == null || centerNode.y == null) return;

    const k = 1.0;
    const tx = width / 2 - centerNode.x * k;
    const ty = height / 2 - centerNode.y * k;
    const t = d3.zoomIdentity.translate(tx, ty).scale(k);

    zoomTransformRef.current = t; // keep in sync
    svg.transition().duration(450).call((zoom as any).transform, t);
  }, [centerId, mode, width, height]);

  /* --------------------------- Render --------------------------- */

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        height: '100%',
        position: 'relative',
        backgroundColor: '#f7fafc'
      }}
    >
      <svg ref={svgRef} width={width} height={height} />
    </div>
  );
}
