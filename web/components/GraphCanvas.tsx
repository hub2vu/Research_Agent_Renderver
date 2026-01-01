/**
 * GraphCanvas Component
 *
 * D3.js force-directed graph visualization.
 *
 * [FIXED - click jump]
 * - Do NOT rebuild the whole graph when `selectedNodeId` changes.
 * - Rebuild graph only when nodes/edges/layout params change.
 * - Update highlight styles in a separate effect.
 *
 * [UPDATED - custom node color]
 * - Supports nodeColorMap override (set from SidePanel)
 * - Keeps selected highlight via stroke/radius (does NOT overwrite fill)
 *
 * [FIXED]
 * - Center flag compatibility: supports both `is_center` and `isCenter`
 * - Cluster color index precedence bug
 */

import React, { useEffect, useRef, useCallback, useState } from 'react';
import * as d3 from 'd3';
import { GraphNode, GraphEdge } from '../lib/mcp';

interface GraphCanvasProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  mode: 'global' | 'paper';
  centerId?: string;
  selectedNodeId?: string;

  // ✅ NEW: node color override map (nodeId -> hex color)
  nodeColorMap?: Record<string, string>;

  onNodeClick?: (node: GraphNode) => void;
  onNodeDoubleClick?: (node: GraphNode) => void;
}

/* -------------------------- Constants --------------------------- */

const CLUSTER_COLORS = [
  '#4299e1', '#48bb78', '#ed8936', '#9f7aea',
  '#f56565', '#38b2ac', '#ed64a6', '#667eea'
];

// Helpers: tolerate both snake_case (backend) and camelCase (frontend)
// + fallback to centerId match (안에 플래그가 없을 수도 있으니)
const makeIsCenterNode = (centerId?: string) => (d: any): boolean =>
  Boolean((d as any).is_center || (d as any).isCenter) || (centerId ? d?.id === centerId : false);

// Safe cluster color index: handles undefined/null/negative and fixes operator precedence bug
const clusterColorIndex = (d: any): number => {
  const n = CLUSTER_COLORS.length;
  const c = Number((d as any).cluster ?? 0);
  return ((c % n) + n) % n;
};

// Basic color validation (#RRGGBB)
const isHexColor = (v: unknown): v is string =>
  typeof v === 'string' && /^#[0-9a-fA-F]{6}$/.test(v);

export default function GraphCanvas({
  nodes,
  edges,
  mode,
  centerId,
  selectedNodeId,
  nodeColorMap,
  onNodeClick,
  onNodeDoubleClick
}: GraphCanvasProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const simulationRef = useRef<d3.Simulation<any, any> | null>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  // D3 Zoom 객체 저장
  const zoomBehaviorRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const zoomSelectionRef = useRef<d3.Selection<SVGSVGElement, unknown, null, undefined> | null>(null);

  // ✅ 노드 selection을 저장해두고, selectedNodeId/nodeColorMap 변경 때 스타일만 업데이트
  const nodeSelectionRef = useRef<d3.Selection<SVGGElement, any, SVGGElement, unknown> | null>(null);

  // Responsive sizing
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const { clientWidth, clientHeight } = containerRef.current;
        setDimensions({
          width: clientWidth || 800,
          height: clientHeight || 600
        });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  const { width, height } = dimensions;

  /* ------------------------ Force Params ------------------------ */

  const getForceParams = useCallback(() => {
    if (mode === 'global') {
      return {
        linkDistance: 60,
        chargeStrength: -300,
        centerStrength: 0.05
      };
    }
    return {
      linkDistance: 80,
      chargeStrength: -500,
      centerStrength: 0.08
    };
  }, [mode]);

  /* ----------------------- Color resolver ----------------------- */

  const getNodeFill = useCallback((d: any) => {
    const isCenterNode = makeIsCenterNode(centerId);

    // Center color stays fixed (원하면 이것도 override 허용 가능)
    if (isCenterNode(d)) return '#f56565';

    // ✅ User override color first
    const override = nodeColorMap?.[d.id];
    if (isHexColor(override)) return override;

    // Default: cluster color
    return CLUSTER_COLORS[clusterColorIndex(d)];
  }, [centerId, nodeColorMap]);

  /* ---------------------- Highlight updater ---------------------- */

  const updateHighlight = useCallback(() => {
    const sel = nodeSelectionRef.current;
    if (!sel) return;

    const isCenterNode = makeIsCenterNode(centerId);

    sel.select('circle')
      .attr('r', (d: any) => {
        if (isCenterNode(d)) return 20;
        if (d.id === selectedNodeId) return 18;
        return 12;
      })
      .attr('fill', (d: any) => {
        // ✅ Do NOT paint selected as yellow; keep real fill color (override/cluster)
        return getNodeFill(d);
      })
      .attr('stroke', (d: any) => (d.id === selectedNodeId ? '#000' : '#fff'))
      .attr('stroke-width', 2);
  }, [selectedNodeId, centerId, getNodeFill]);

  /* -------------------------- Main D3 --------------------------- */

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);

    // Initialize or restore zoom behavior
    if (!zoomBehaviorRef.current) {
      const zoom = d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.1, 4])
        .on('zoom', (event) => {
          svg.select('g.canvas-root').attr('transform', event.transform.toString());
        });

      zoomBehaviorRef.current = zoom;
      zoomSelectionRef.current = svg;
      svg.call(zoom);
    }

    // Clear old graph (but preserve zoom behavior)
    // ✅ NOTE: this effect no longer runs on `selectedNodeId` / `nodeColorMap` change, so click/color won't rebuild.
    svg.selectAll('g.canvas-root').remove();

    const container = svg.append('g').attr('class', 'canvas-root');

    // Convert to D3 mutable objects
    const graphNodes = nodes.map(n => ({ ...n }));
    const graphEdges = edges.map(e => ({ ...e }));

    const { linkDistance, chargeStrength, centerStrength } = getForceParams();

    const isCenterNode = makeIsCenterNode(centerId);

    // Simulation
    const simulation = d3.forceSimulation(graphNodes as any)
      .force('link', d3.forceLink(graphEdges as any).id((d: any) => d.id).distance(linkDistance))
      .force('charge', d3.forceManyBody().strength(chargeStrength))
      // ⚠️ forceCenter는 d3 버전에 따라 strength()가 없을 수 있음.
      .force(
        'center',
        (d3.forceCenter(width / 2, height / 2) as any).strength?.(centerStrength) ??
          d3.forceCenter(width / 2, height / 2)
      )
      .force('collide', d3.forceCollide().radius(30));

    simulationRef.current = simulation;

    if (mode === 'paper' && centerId) {
      const centerNode = graphNodes.find(n => n.id === centerId);
      if (centerNode) {
        (centerNode as any).fx = width / 2;
        (centerNode as any).fy = height / 2;
      }
    }

    // Edges
    const link = container.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(graphEdges)
      .enter()
      .append('line')
      .attr('stroke', '#cbd5e0')
      .attr('stroke-opacity', 0.6)
      .attr('stroke-width', (d: any) => (d.type === 'references' ? 1 : 2));

    // Nodes
    const node = container.append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(graphNodes)
      .enter()
      .append('g')
      .style('cursor', 'pointer')
      .call(d3.drag<any, any>()
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

    // ✅ store node selection for highlight-only updates
    nodeSelectionRef.current = node as any;

    // Circle
    node.append('circle')
      .attr('r', (d: any) => {
        if (isCenterNode(d)) return 20;
        if (d.id === selectedNodeId) return 18;
        return 12;
      })
      .attr('fill', (d: any) => getNodeFill(d))
      .attr('stroke', (d: any) => (d.id === selectedNodeId ? '#000' : '#fff'))
      .attr('stroke-width', 2);

    // Labels
    node.append('text')
      .attr('dy', 25)
      .attr('text-anchor', 'middle')
      .attr('font-size', '10px')
      .attr('fill', '#4a5568')
      .text((d: any) => {
        const title = d.title || d.id;
        return title.length > 20 ? title.slice(0, 20) + '...' : title;
      });

    // Events
    node.on('click', (event, d: any) => {
      event.stopPropagation();
      onNodeClick?.(d as GraphNode);
    });

    node.on('dblclick', (event, d: any) => {
      event.stopPropagation();
      onNodeDoubleClick?.(d as GraphNode);
    });

    // Tick
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
    getNodeFill
    // ✅ selectedNodeId/nodeColorMap 제거: 클릭/색 변경으로 그래프 재생성 금지
  ]);

  /* ------------------- Highlight-only effect ------------------- */
  useEffect(() => {
    updateHighlight();
  }, [updateHighlight]);

  /* ---------------- [Auto-Zoom to Center] ---------------- */
  useEffect(() => {
    if (!centerId || mode !== 'paper' || !simulationRef.current || !zoomBehaviorRef.current || !zoomSelectionRef.current) {
      return;
    }

    const sim = simulationRef.current;
    const svg = zoomSelectionRef.current;
    const zoom = zoomBehaviorRef.current;

    const centerNode: any = (sim.nodes() as any[]).find(n => n.id === centerId);
    if (!centerNode || centerNode.x == null || centerNode.y == null) return;

    const k = 1.0; // keep scale
    const tx = width / 2 - centerNode.x * k;
    const ty = height / 2 - centerNode.y * k;
    const t = d3.zoomIdentity.translate(tx, ty).scale(k);

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
