import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { getDocument, GlobalWorkerOptions } from 'pdfjs-dist';
import workerSrc from 'pdfjs-dist/build/pdf.worker.min.mjs?url';

// Vite 호환 worker 설정
GlobalWorkerOptions.workerSrc = workerSrc;

type PdfDoc = {
  numPages: number;
  getPage: (pageNumber: number) => Promise<any>;
  destroy?: () => Promise<void> | void;
};

type PdfLoadResult = {
  usedId: string;
  url: string;
  doc: PdfDoc;
};

function stripPrefixes(id: string): string {
  return String(id ?? '').replace(/^(paper:|ref:)/i, '').trim();
}

function normalizeArxivToDoiLike(id: string): string {
  if (id.startsWith('10.48550_arxiv.')) return id;
  const m = id.match(/^(\d{4}\.\d{4,5})(v\d+)?$/);
  if (m) return `10.48550_arxiv.${m[1]}`;
  return id;
}

async function resolveAndLoadPdf(paperIdRaw: string): Promise<PdfLoadResult> {
  const cleaned = stripPrefixes(paperIdRaw);
  const candidates = [cleaned, normalizeArxivToDoiLike(cleaned)].filter((v, i, a) => a.indexOf(v) === i);

  const tried: string[] = [];
  const errors: string[] = [];

  for (const id of candidates) {
    const urlCandidates = [
      `/output/${id}/paper.pdf`,
      `/output/${id}/${id}.pdf`,
      `/output/${id}/main.pdf`,
      `/pdf/${id}.pdf`,
      `/pdf/neurips2025/${id}.pdf`
    ];

    for (const url of urlCandidates) {
      tried.push(url);
      try {
        const loadingTask: any = getDocument(url);
        const doc: PdfDoc = await loadingTask.promise;
        return { usedId: id, url, doc };
      } catch (err) {
        errors.push(`${url} (${String(err)})`);
      }
    }
  }

  throw new Error(
    [
      'PDF를 불러올 수 없습니다.',
      '',
      '확인해보세요:',
      '- /output/{paperId}/paper.pdf 또는 /pdf/{paperId}.pdf 존재 여부',
      '- noteId(paperId)가 폴더명/파일명과 정확히 일치하는지 확인',
      '',
      `시도한 후보 URL: ${tried.join(' | ')}`,
      errors.length ? '' : '',
      errors.length ? `에러 로그:\n${errors.join('\n')}` : ''
    ].join('\n')
  );
}

export default function NotePage(props: { noteId?: string } = {}) {
  const params = useParams();

  const rawFromRoute = (params as any).paperId ?? '';
  const raw = props.noteId ?? rawFromRoute ?? '';
  const paperId = decodeURIComponent(String(raw));

  const navigate = useNavigate();

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [usedId, setUsedId] = useState<string>('');
  const [pdfUrl, setPdfUrl] = useState<string>('');
  const [pdfDoc, setPdfDoc] = useState<PdfDoc | null>(null);
  const [pageCount, setPageCount] = useState<number>(0);

  const canvasRefs = useRef<Array<HTMLCanvasElement | null>>([]);
  const pdfDocRef = useRef<PdfDoc | null>(null);

  // 노트(우측) - 로컬 저장(즉시 저장)
  const noteStorageKey = useMemo(() => `note:${usedId || stripPrefixes(paperId)}`, [usedId, paperId]);
  const [note, setNote] = useState<string>(() => {
    try {
      return localStorage.getItem(noteStorageKey) ?? '';
    } catch {
      return '';
    }
  });

  useEffect(() => {
    try {
      const v = localStorage.getItem(noteStorageKey) ?? '';
      setNote(v);
    } catch {
      setNote('');
    }
  }, [noteStorageKey]);

  useEffect(() => {
    try {
      localStorage.setItem(noteStorageKey, note);
    } catch {
      // ignore
    }
  }, [note, noteStorageKey]);

  useEffect(() => {
    const cleaned = stripPrefixes(paperId);

    if (!cleaned) {
      setError('paperId가 비어 있습니다. 라우트 파라미터(/note/:paperId)를 확인하세요.');
      setLoading(false);
      return;
    }

    let cancelled = false;

    const load = async () => {
      setLoading(true);
      setError(null);

      // 이전 doc 정리
      try {
        if (pdfDocRef.current?.destroy) await pdfDocRef.current.destroy();
      } catch {
        // ignore
      }
      pdfDocRef.current = null;
      setPdfDoc(null);
      setPageCount(0);
      setPdfUrl('');
      setUsedId('');

      try {
        const res = await resolveAndLoadPdf(cleaned);
        if (cancelled) {
          try { if (res.doc.destroy) await res.doc.destroy(); } catch {}
          return;
        }
        pdfDocRef.current = res.doc;
        setUsedId(res.usedId);
        setPdfUrl(res.url);
        setPdfDoc(res.doc);
        setPageCount(res.doc.numPages);
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e));
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    load();

    return () => {
      cancelled = true;
      try {
        if (pdfDocRef.current?.destroy) pdfDocRef.current.destroy();
      } catch {
        // ignore
      }
    };
  }, [paperId]);

  const pages = useMemo(() => Array.from({ length: pageCount }, (_, i) => i + 1), [pageCount]);

  useEffect(() => {
    if (!pdfDoc || pageCount <= 0) return;

    let cancelled = false;

    const renderPages = async () => {
      for (const pageNumber of pages) {
        if (cancelled) return;

        const page = await pdfDoc.getPage(pageNumber);
        const viewport = page.getViewport({ scale: 1.2 });

        const canvas = canvasRefs.current[pageNumber - 1];
        if (!canvas) continue;

        const ctx = canvas.getContext('2d');
        if (!ctx) continue;

        const outputScale = window.devicePixelRatio || 1;

        canvas.width = Math.floor(viewport.width * outputScale);
        canvas.height = Math.floor(viewport.height * outputScale);
        canvas.style.width = `${Math.floor(viewport.width)}px`;
        canvas.style.height = `${Math.floor(viewport.height)}px`;

        const transform =
          outputScale !== 1 ? ([outputScale, 0, 0, outputScale, 0, 0] as [number, number, number, number, number, number]) : undefined;

        await page.render({ canvasContext: ctx, viewport, transform }).promise;
      }
    };

    renderPages().catch(() => {});

    return () => {
      cancelled = true;
    };
  }, [pdfDoc, pages, pageCount]);

  return (
    <div style={{ display: 'flex', height: '100vh', backgroundColor: '#f5f5f5' }}>
      {/* Left: PDF paper view */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', borderRight: '1px solid #e2e8f0' }}>
        <header
          style={{
            padding: '16px 20px',
            backgroundColor: '#fff',
            borderBottom: '1px solid #e2e8f0',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}
        >
          <div>
            <button
              onClick={() => navigate(-1)}
              style={{
                padding: '6px 12px',
                borderRadius: '6px',
                backgroundColor: '#edf2f7',
                border: 'none',
                cursor: 'pointer',
                marginRight: '12px'
              }}
            >
              ← Back
            </button>
            <span style={{ fontWeight: 700 }}>Paper View</span>
          </div>

          <div style={{ fontSize: 12, color: '#718096', textAlign: 'right' }}>
            <div>noteId: {paperId}</div>
            {usedId && <div>usedId: {usedId}</div>}
            {pdfUrl && <div>pdf source: {pdfUrl}</div>}
            {pageCount > 0 && <div>pages: {pageCount}</div>}
          </div>
        </header>

        <div style={{ flex: 1, overflowY: 'auto', padding: '14px 18px' }}>
          {loading && <div style={{ color: '#718096', fontSize: 14, padding: '12px 0' }}>PDF를 불러오는 중…</div>}

          {error && (
            <pre
              style={{
                background: '#fff',
                border: '1px solid #fed7d7',
                color: '#c53030',
                padding: 12,
                borderRadius: 8,
                whiteSpace: 'pre-wrap'
              }}
            >
              {error}
            </pre>
          )}

          {!loading && !error && pageCount === 0 && (
            <div style={{ color: '#718096', fontSize: 14 }}>
              PDF 페이지를 찾을 수 없습니다. (/output/{'{paperId}'}/paper.pdf 또는 /pdf/{'{paperId}'}.pdf 확인)
            </div>
          )}

          {!loading && !error && pageCount > 0 && (
            <div style={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 10, padding: 16 }}>
              {pages.map((pageNumber) => (
                <div key={`page-${pageNumber}`} style={{ marginBottom: 18 }}>
                  <div style={{ fontWeight: 700, margin: '8px 0 10px', color: '#2d3748' }}>Page {pageNumber}</div>
                  <div style={{ display: 'flex', justifyContent: 'center' }}>
                    <canvas
                      ref={(el) => {
                        canvasRefs.current[pageNumber - 1] = el;
                      }}
                      style={{
                        maxWidth: '100%',
                        borderRadius: 8,
                        border: '1px solid #edf2f7',
                        background: '#fff'
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Right: Note panel */}
      <div style={{ width: '40%', minWidth: 360, display: 'flex', flexDirection: 'column' }}>
        <header
          style={{
            padding: '16px 20px',
            backgroundColor: '#fff',
            borderBottom: '1px solid #e2e8f0'
          }}
        >
          <div style={{ fontWeight: 700 }}>논문 분석 노트</div>
        </header>

        <div style={{ flex: 1, padding: 14, backgroundColor: '#f5f5f5' }}>
          <textarea
            value={note}
            onChange={(e) => setNote(e.target.value)}
            placeholder="여기에 논문 분석/메모를 작성하세요…"
            style={{
              width: '100%',
              height: '100%',
              resize: 'none',
              borderRadius: 10,
              border: '1px solid #e2e8f0',
              padding: 12,
              fontSize: 14,
              lineHeight: 1.5,
              outline: 'none'
            }}
          />
        </div>
      </div>
    </div>
  );
}
