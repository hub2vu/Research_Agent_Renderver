/**
 * GraphCanvas Component
 *
 * D3.js force-directed graph visualization.
 * [UPDATED] Fixes node position reset & camera jump on re-renders.
 *
 * [FIXED]
 * - Center flag compatibility: supports both `is_center` (snake_case) and `isCenter` (camelCase)
 * - Cluster color index precedence bug: fixes `d.cluster || 0 % ...` → proper modulo
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
  onNodeClick?: (node: GraphNode) => void;
  onNodeDoubleClick?: (node: GraphNode) => void;
}

/* -------------------------- Constants --------------------------- */

const CLUSTER_COLORS = [
  '#4299e1', '#48bb78', '#ed8936', '#9f7aea',
  '#f56565', '#38b2ac', '#ed64a6', '#667eea'
];

// Helpers: tolerate both snake_case (backend) and camelCase (frontend)
const isCenterNode = (d: any): boolean => Boolean((d as any).is_center || (d as any).isCenter);

// Safe cluster color index: handles undefined/null/negative and fixes operator precedence bug
const clusterColorIndex = (d: any): number => {
  const n = CLUSTER_COLORS.length;
  const c = Number((d as any).cluster ?? 0);
  return ((c % n) + n) % n;
};

export default function GraphCanvas({
  nodes,
  edges,
  mode,
  centerId,
  selectedNodeId,
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

  // Responsive sizing
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const { clientWidth, clientHeight } = containerRef.current;
        setDimensions({ width: clientWidth, height: clientHeight });
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

  /* -------------------------- Main D3 --------------------------- */

  useEffect(() => {
    if (!svgRef.current) return;

    // Save/restore zoom state to avoid camera jump
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
    svg.selectAll('g.canvas-root').remove();

    const container = svg.append('g').attr('class', 'canvas-root');

    // Convert to D3 mutable objects
    const graphNodes = nodes.map(n => ({ ...n }));
    const graphEdges = edges.map(e => ({ ...e }));

    const { linkDistance, chargeStrength, centerStrength } = getForceParams();

    // Simulation
    const simulation = d3.forceSimulation(graphNodes as any)
      .force('link', d3.forceLink(graphEdges as any).id((d: any) => d.id).distance(linkDistance))
      .force('charge', d3.forceManyBody().strength(chargeStrength))
      .force('center', d3.forceCenter(width / 2, height / 2).strength(centerStrength))
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
      .attr('stroke-width', d => (d.type === 'references' ? 1 : 2));

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

    // Circle
    node.append('circle')
      .attr('r', (d: any) => {
        if (isCenterNode(d)) return 20;
        if (d.id === selectedNodeId) return 18; // Highlight selected
        return 12;
      })
      .attr('fill', (d: any) => {
        if (isCenterNode(d)) return '#f56565';
        if (d.id === selectedNodeId) return '#ecc94b'; // Highlight selected
        return CLUSTER_COLORS[clusterColorIndex(d)];
      })
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
      if (onNodeClick) onNodeClick(d as GraphNode);
    });

    node.on('dblclick', (event, d: any) => {
      event.stopPropagation();
      if (onNodeDoubleClick) onNodeDoubleClick(d as GraphNode);
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
    selectedNodeId
  ]);

  /* ---------------- [NEW] Auto-Zoom to Center ---------------- */
  useEffect(() => {
    // centerId가 바뀔 때만 줌 동작 수행 (초기 렌더링 시에는 위에서 복구한 줌 상태 유지)
    if (!centerId || mode !== 'paper' || !simulationRef.current || !zoomBehaviorRef.current || !zoomSelectionRef.current) {
      return;
    }

    const sim = simulationRef.current;
    const svg = zoomSelectionRef.current;
    const zoom = zoomBehaviorRef.current;

    const centerNode: any = (sim.nodes() as any[]).find(n => n.id === centerId);
    if (!centerNode || centerNode.x == null || centerNode.y == null) return;

    // center node가 화면 중앙에 오도록 transform 계산
    const k = 1.0; // keep scale
    const tx = width / 2 - centerNode.x * k;
    const ty = height / 2 - centerNode.y * k;
    const t = d3.zoomIdentity.translate(tx, ty).scale(k);

    svg.transition().duration(450).call(zoom.transform as any, t);
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
