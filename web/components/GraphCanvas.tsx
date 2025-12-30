/**
 * GraphCanvas Component
 *
 * D3.js force-directed graph visualization.
 * [UPDATED] Fixes node position reset & camera jump on re-renders.
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

const CLUSTER_COLORS = [
  '#4299e1', '#48bb78', '#ed8936', '#9f7aea',
  '#f56565', '#38b2ac', '#ed64a6', '#667eea'
];

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
        setDimensions({ width: clientWidth || 800, height: clientHeight || 600 });
      }
    };
    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  const { width, height } = dimensions;

  // Get force parameters
  const getForceParams = useCallback(() => {
    if (mode === 'paper') {
      return { charge: -300, linkDistance: 100, centerForce: 0.1 };
    } else {
      return { charge: -500, linkDistance: 150, centerForce: 0.05 };
    }
  }, [mode]);

  // Initialize graph
  useEffect(() => {
    if (!svgRef.current || nodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    const params = getForceParams();

    // [FIX 1] 기존 시뮬레이션에서 노드 좌표 백업 (위치 초기화 방지)
    const oldNodes = new Map(simulationRef.current?.nodes().map((n: any) => [n.id, n]) || []);
    
    // [FIX 2] 기존 줌 상태 백업 (카메라 튀는 현상 방지)
    const oldTransform = d3.zoomTransform(svg.node()!);

    // 캔버스 초기화
    svg.selectAll('*').remove();

    const container = svg.append('g');

    // Zoom setup
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        container.attr('transform', event.transform);
      });

    svg.call(zoom);
    
    // [FIX 2-1] 백업해둔 줌 상태 복구
    if (oldTransform && oldTransform.k !== 0) { // k=0 check to avoid invalid transform
        svg.call(zoom.transform, oldTransform);
    }
    
    zoomBehaviorRef.current = zoom;
    zoomSelectionRef.current = svg;

    // Prepare data
    // [FIX 1-1] 새 노드 생성 시 기존 좌표(x, y, vx, vy) 계승
    const graphNodes = nodes.map(n => {
        const oldNode = oldNodes.get(n.id);
        if (oldNode) {
            return { 
                ...n, 
                x: oldNode.x, 
                y: oldNode.y,
                vx: oldNode.vx,
                vy: oldNode.vy,
                fx: oldNode.fx, // 고정된 상태도 유지
                fy: oldNode.fy
            };
        }
        return { ...n };
    });

    const nodeMap = new Map(graphNodes.map(n => [n.id, n]));
    
    const graphEdges = edges
      .filter(e => nodeMap.has(e.source as string) && nodeMap.has(e.target as string))
      .map(e => ({
        source: e.source,
        target: e.target,
        weight: e.weight || 1,
        type: e.type
      }));

    // Simulation
    const simulation = d3.forceSimulation(graphNodes as any)
      .force('link', d3.forceLink(graphEdges as any)
        .id((d: any) => d.id)
        .distance(params.linkDistance)
        .strength((d: any) => d.weight || 0.5)
      )
      .force('charge', d3.forceManyBody().strength(params.charge))
      .force('center', d3.forceCenter(width / 2, height / 2).strength(params.centerForce))
      .force('collision', d3.forceCollide().radius(30));

    // [FIX 3] 알파(Alpha) 값 조정
    // 노드 위치를 복구했으므로, 시뮬레이션을 처음부터(alpha=1) 강하게 돌릴 필요가 없음.
    // 기존 위치가 있다면 alpha를 낮게 시작하여 부드럽게 안정화.
    if (oldNodes.size > 0) {
        simulation.alpha(0.3).restart();
    }

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
      .attr('stroke', d => d.type === 'similarity' ? '#a0aec0' : '#718096')
      .attr('stroke-opacity', d => d.type === 'similarity' ? d.weight : 0.6)
      .attr('stroke-width', d => d.type === 'similarity' ? d.weight * 2 : 1.5);

    // Nodes
    const node = container.append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(graphNodes)
      .enter()
      .append('g')
      .attr('class', 'node')
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
      .attr('r', d => {
        if (d.is_center) return 20;
        if (d.id === selectedNodeId) return 18; // Highlight selected
        return 12;
      })
      .attr('fill', d => {
        if (d.is_center) return '#f56565';
        if (d.id === selectedNodeId) return '#ecc94b'; // Highlight selected
        return CLUSTER_COLORS[d.cluster || 0 % CLUSTER_COLORS.length];
      })
      .attr('stroke', d => d.id === selectedNodeId ? '#000' : '#fff')
      .attr('stroke-width', 2);

    // Labels
    node.append('text')
      .attr('dy', 25)
      .attr('text-anchor', 'middle')
      .attr('font-size', '10px')
      .attr('fill', '#4a5568')
      .text(d => {
        const title = d.title || d.id;
        return title.length > 20 ? title.slice(0, 20) + '...' : title;
      });

    // Events
    node.on('click', (event, d) => {
      event.stopPropagation();
      if (onNodeClick) onNodeClick(d as GraphNode);
    });

    node.on('dblclick', (event, d) => {
      event.stopPropagation();
      if (onNodeDoubleClick) onNodeDoubleClick(d as GraphNode);
    });

    // Tick
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      node.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
    });

    return () => {
      simulation.stop();
    };
  }, [nodes, edges, mode, centerId, width, height, getForceParams, onNodeClick, onNodeDoubleClick, selectedNodeId]);


  /* ---------------- [NEW] Auto-Zoom to Center ---------------- */
  useEffect(() => {
    // centerId가 바뀔 때만 줌 동작 수행 (초기 렌더링 시에는 위에서 복구한 줌 상태 유지)
    if (!centerId || mode !== 'paper' || !simulationRef.current || !zoomBehaviorRef.current || !zoomSelectionRef.current) {
        return;
    }
    
    // 만약 centerId가 변경된 것이 아니라 단순히 리렌더링 된 것이라면 줌 동작을 스킵하는 로직을 추가할 수도 있습니다.
    // 여기서는 간단히 타이머를 두고 실행합니다.

    const timer = setTimeout(() => {
        const nodes = simulationRef.current?.nodes() as any[];
        const targetNode = nodes.find(n => n.id === centerId);

        if (targetNode && targetNode.x !== undefined) {
             const zoomIdentity = d3.zoomIdentity
                .translate(width / 2, height / 2)
                .scale(1.0)
                .translate(-targetNode.x, -targetNode.y);

             zoomSelectionRef.current?.transition()
                .duration(750)
                .call(zoomBehaviorRef.current.transform, zoomIdentity);
        }
    }, 500);

    return () => clearTimeout(timer);
  }, [centerId, mode, width, height]);


  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        height: '100%',
        border: '1px solid #e2e8f0',
        borderRadius: '8px',
        backgroundColor: '#f7fafc',
        overflow: 'hidden'
      }}
    >
      <svg
        ref={svgRef}
        width={width}
        height={height}
      />
    </div>
  );
}