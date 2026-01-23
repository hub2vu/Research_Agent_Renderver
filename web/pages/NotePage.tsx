import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';

type ImageMetaItem = {
  page_number: number;
  image_index: number;
  filename: string;
  path?: string;
  format?: string;
  size_bytes?: number;
};

type ImageMetaJson = {
  filename?: string;
  total_pages?: number;
  images?: ImageMetaItem[];
  total_images_extracted?: number;
};

type LoadResult = {
  usedId: string;
  text: string;
  textSource: 'txt' | 'json';
  imageMeta?: ImageMetaJson | null;
};

function normalizeArxivToDoiLike(id: string): string {
  if (id.startsWith('10.48550_arxiv.')) return id;
  const m = id.match(/^(\d{4}\.\d{4,5})(v\d+)?$/);
  if (m) return `10.48550_arxiv.${m[1]}`;
  return id;
}

async function fetchTextFromFolder(folderId: string): Promise<{ ok: boolean; text?: string; source?: 'txt' | 'json'; status?: number }> {
  // 1) txt 우선
  try {
    const resTxt = await fetch(`/output/${folderId}/extracted_text.txt`, { cache: 'no-store' });
    if (resTxt.ok) {
      const text = await resTxt.text();
      return { ok: true, text, source: 'txt', status: resTxt.status };
    }
    // 2) json fallback
    const resJson = await fetch(`/output/${folderId}/extracted_text.json`, { cache: 'no-store' });
    if (resJson.ok) {
      const j = await resJson.json();
      const text = typeof j === 'string' ? j : (j?.text ?? j?.content ?? j?.full_text ?? '');
      if (typeof text === 'string' && text.trim().length > 0) {
        return { ok: true, text, source: 'json', status: resJson.status };
      }
      return { ok: false, status: resJson.status };
    }
    return { ok: false, status: resTxt.status || resJson.status };
  } catch {
    return { ok: false };
  }
}

async function fetchImageMeta(folderId: string): Promise<ImageMetaJson | null> {
  try {
    const res = await fetch(`/output/${folderId}/image_metadata.json`, { cache: 'no-store' });
    if (!res.ok) return null;
    const json = (await res.json()) as ImageMetaJson;
    return json;
  } catch {
    return null;
  }
}

async function resolveAndLoad(paperId: string): Promise<LoadResult> {
  const raw = paperId.trim();
  const candidates = [raw, normalizeArxivToDoiLike(raw)].filter((v, i, a) => a.indexOf(v) === i);

  const tried: string[] = [];

  for (const id of candidates) {
    tried.push(id);

    const textRes = await fetchTextFromFolder(id);
    if (textRes.ok && textRes.text && textRes.source) {
      const imageMeta = await fetchImageMeta(id);
      return {
        usedId: id,
        text: textRes.text,
        textSource: textRes.source,
        imageMeta
      };
    }
  }

  throw new Error(
    [
      '폴더를 출력할 수 없습니다.',
      '',
      '확인해보세요:',
      '- /output/{paperId}/extracted_text.txt 또는 extracted_text.json 존재 여부',
      '- paperId가 폴더명과 정확히 일치하는지 확인',
      '',
      `시도한 후보 paperId: ${tried.join(' | ')}`
    ].join('\n')
  );
}

function splitByPages(extractedText: string): Array<{ page: number | null; lines: string[] }> {
  const lines = extractedText.replace(/\r\n/g, '\n').split('\n');

  const blocks: Array<{ page: number | null; lines: string[] }> = [];
  let currentPage: number | null = null;
  let buf: string[] = [];

  const flush = () => {
    if (buf.length > 0) blocks.push({ page: currentPage, lines: buf });
    buf = [];
  };

  for (const line of lines) {
    const m = line.match(/^===\s*Page\s+(\d+)\s*===$/);
    if (m) {
      flush();
      currentPage = parseInt(m[1], 10);
      // 페이지 헤더는 별도로 렌더링할 거라 본문에는 넣지 않음
      continue;
    }
    buf.push(line);
  }
  flush();

  // 헤더가 전혀 없으면 한 덩어리로
  if (blocks.length === 1 && blocks[0].page === null) return blocks;
  return blocks;
}

export default function NotePage() {
  const { paperId: rawId } = useParams();
  const paperId = rawId ? decodeURIComponent(rawId) : '';
  const navigate = useNavigate();

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [usedId, setUsedId] = useState<string>('');
  const [text, setText] = useState<string>('');
  const [textSource, setTextSource] = useState<'txt' | 'json' | ''>('');
  const [imageMeta, setImageMeta] = useState<ImageMetaJson | null>(null);

  // 노트(우측) - 로컬 저장(즉시 저장)
  const noteStorageKey = useMemo(() => `note:${usedId || paperId}`, [usedId, paperId]);
  const [note, setNote] = useState<string>(() => {
    try {
      return localStorage.getItem(noteStorageKey) ?? '';
    } catch {
      return '';
    }
  });

  useEffect(() => {
    // usedId가 바뀌면 해당 키에서 다시 로드
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

  const load = useCallback(async () => {
    if (!paperId) return;
    setLoading(true);
    setError(null);

    try {
      const res = await resolveAndLoad(paperId);
      setUsedId(res.usedId);
      setText(res.text);
      setTextSource(res.textSource);
      setImageMeta(res.imageMeta ?? null);
      setLoading(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setLoading(false);
    }
  }, [paperId]);

  useEffect(() => {
    load();
  }, [load]);

  const imagesByPage = useMemo(() => {
    const map = new Map<number, ImageMetaItem[]>();
    const imgs = imageMeta?.images ?? [];
    for (const it of imgs) {
      if (!map.has(it.page_number)) map.set(it.page_number, []);
      map.get(it.page_number)!.push(it);
    }
    // page 내부 이미지 index 순 정렬
    for (const [k, arr] of map.entries()) {
      arr.sort((a, b) => (a.image_index ?? 0) - (b.image_index ?? 0));
      map.set(k, arr);
    }
    return map;
  }, [imageMeta]);

  const blocks = useMemo(() => splitByPages(text), [text]);

  return (
    <div style={{ display: 'flex', height: '100vh', backgroundColor: '#f5f5f5' }}>
      {/* Left: Extracted paper view */}
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
            <div>paperId: {paperId}</div>
            {usedId && <div>folder used: {usedId}</div>}
            {textSource && <div>text source: extracted_text.{textSource}</div>}
            {imageMeta?.total_images_extracted != null && <div>images: {imageMeta.total_images_extracted}</div>}
          </div>
        </header>

        <div style={{ flex: 1, overflowY: 'auto', padding: '14px 18px' }}>
          {loading && (
            <div style={{ color: '#718096', fontSize: 14, padding: '12px 0' }}>
              추출된 텍스트를 불러오는 중…
            </div>
          )}

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

          {!loading && !error && text.trim().length === 0 && (
            <div style={{ color: '#718096', fontSize: 14 }}>
              추출 텍스트가 비어 있습니다. (/output/{'{paperId}'}/extracted_text.txt 또는 json 확인)
            </div>
          )}

          {!loading && !error && text.trim().length > 0 && (
            <div style={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 10, padding: 16 }}>
              {blocks.map((blk, idx) => {
                const page = blk.page;
                const imgs = page != null ? imagesByPage.get(page) ?? [] : [];

                return (
                  <div key={`${page ?? 'nopage'}-${idx}`} style={{ marginBottom: 18 }}>
                    {page != null && (
                      <div style={{ fontWeight: 700, margin: '8px 0 10px', color: '#2d3748' }}>
                        Page {page}
                      </div>
                    )}

                    {/* page header 아래에 해당 페이지 이미지 삽입 */}
                    {imgs.length > 0 && (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginBottom: 12 }}>
                        {imgs.map((im) => (
                          <img
                            key={`${im.page_number}-${im.image_index}-${im.filename}`}
                            src={`/output/${usedId}/images/${im.filename}`}
                            alt={im.filename}
                            style={{
                              maxWidth: '100%',
                              borderRadius: 8,
                              border: '1px solid #edf2f7'
                            }}
                            loading="lazy"
                          />
                        ))}
                      </div>
                    )}

                    {/* text */}
                    <pre
                      style={{
                        whiteSpace: 'pre-wrap',
                        fontSize: 13,
                        lineHeight: 1.6,
                        color: '#2d3748',
                        margin: 0
                      }}
                    >
                      {blk.lines.join('\n')}
                    </pre>
                  </div>
                );
              })}
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
              padding: 14,
              fontSize: 14,
              lineHeight: 1.6,
              outline: 'none'
            }}
          />
        </div>
      </div>
    </div>
  );
}
