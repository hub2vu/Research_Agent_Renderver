/**
 * GraphCanvas Component
 *
 * D3.js force-directed graph visualization.
 * Handles node rendering, physics simulation, and user interactions.
 *
 * IMPORTANT: This component only handles visualization.
 * All data fetching goes through MCP API.
 */

import React, { useEffect, useRef, useCallback, useState } from 'react';
import * as d3 from 'd3';
import { GraphNode, GraphEdge } from '../api/mcp';

interface GraphCanvasProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  mode: 'global' | 'paper';
  centerId?: string;
  selectedNodeId?: string;
  onNodeClick?: (node: GraphNode) => void;
  onNodeDoubleClick?: (node: GraphNode) => void;
}

// Color palette for clusters
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

  // Get force parameters based on mode
  const getForceParams = useCallback(() => {
    if (mode === 'paper') {
      // Graph A: Center node fixed, strong repulsion
      return {
        charge: -300,
        linkDistance: 100,
        centerForce: 0.1
      };
    } else {
      // Graph B: Very strong repulsion, no center
      return {
        charge: -500,
        linkDistance: 150,
        centerForce: 0.05
      };
    }
  }, [mode]);

  // Initialize or update the graph
  useEffect(() => {
    if (!svgRef.current || nodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    const params = getForceParams();

    // Clear previous content
    svg.selectAll('*').remove();

    // Create container group for zoom
    const container = svg.append('g');

    // Setup zoom
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        container.attr('transform', event.transform);
      });

    svg.call(zoom);

    // Prepare data
    const nodeMap = new Map(nodes.map(n => [n.id, { ...n }]));
    const graphNodes = Array.from(nodeMap.values());
    const graphEdges = edges
      .filter(e => nodeMap.has(e.source as string) && nodeMap.has(e.target as string))
      .map(e => ({
        source: e.source,
        target: e.target,
        weight: e.weight || 1,
        type: e.type
      }));

    // Create force simulation
    const simulation = d3.forceSimulation(graphNodes as any)
      .force('link', d3.forceLink(graphEdges as any)
        .id((d: any) => d.id)
        .distance(params.linkDistance)
        .strength((d: any) => d.weight || 0.5)
      )
      .force('charge', d3.forceManyBody().strength(params.charge))
      .force('center', d3.forceCenter(width / 2, height / 2).strength(params.centerForce))
      .force('collision', d3.forceCollide().radius(30));

    simulationRef.current = simulation;

    // Fix center node in paper mode
    if (mode === 'paper' && centerId) {
      const centerNode = graphNodes.find(n => n.id === centerId);
      if (centerNode) {
        (centerNode as any).fx = width / 2;
        (centerNode as any).fy = height / 2;
      }
    }

    // Draw edges
    const link = container.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(graphEdges)
      .enter()
      .append('line')
      .attr('stroke', d => d.type === 'similarity' ? '#a0aec0' : '#718096')
      .attr('stroke-opacity', d => d.type === 'similarity' ? d.weight : 0.6)
      .attr('stroke-width', d => d.type === 'similarity' ? d.weight * 2 : 1.5);

    // Draw nodes
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
          // Keep center node fixed in paper mode
          if (mode !== 'paper' || d.id !== centerId) {
            d.fx = null;
            d.fy = null;
          }
        })
      );

    // Node circles
    node.append('circle')
      .attr('r', d => {
        if (d.is_center) return 20;
        if (d.depth === 0) return 15;
        return 12;
      })
      .attr('fill', d => {
        if (d.is_center) return '#f56565';
        return CLUSTER_COLORS[d.cluster || 0 % CLUSTER_COLORS.length];
      })
      .attr('stroke', '#fff')
      .attr('stroke-width', 2);

    // Node labels
    node.append('text')
      .attr('dy', 25)
      .attr('text-anchor', 'middle')
      .attr('font-size', '10px')
      .attr('fill', '#4a5568')
      .text(d => {
        const title = d.title || d.id;
        return title.length > 20 ? title.slice(0, 20) + '...' : title;
      });

    // Click handlers
    node.on('click', (event, d) => {
      event.stopPropagation();
      if (onNodeClick) onNodeClick(d as GraphNode);
    });

    node.on('dblclick', (event, d) => {
      event.stopPropagation();
      if (onNodeDoubleClick) onNodeDoubleClick(d as GraphNode);
    });

    // Tick function
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      node.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
    });

    // Cleanup
    return () => {
      simulation.stop();
    };
  }, [nodes, edges, mode, centerId, width, height, getForceParams, onNodeClick, onNodeDoubleClick]);

  // Handle incremental updates
  const addNodes = useCallback((newNodes: GraphNode[], newEdges: GraphEdge[]) => {
    // This would be called for incremental Graph A updates
    // For now, we rely on full re-render via useEffect
  }, []);

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
