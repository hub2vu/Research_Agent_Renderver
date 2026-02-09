/**
 * ConferenceInfoModal
 *
 * NeurIPS / ICLR 페이지 진입 시 처음 한 번 표시되는 안내 모달.
 * localStorage에 dismiss 상태를 저장하여 다시 표시하지 않음.
 * "다시 보지 않기" 체크 없이, X 또는 확인 버튼으로 닫으면 세션 동안 다시 뜨지 않음.
 */

import React, { useState, useEffect } from 'react';

const STORAGE_KEY = 'conference_info_dismissed';

export default function ConferenceInfoModal() {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const dismissed = sessionStorage.getItem(STORAGE_KEY);
    if (!dismissed) {
      setVisible(true);
    }
  }, []);

  const handleClose = () => {
    setVisible(false);
    sessionStorage.setItem(STORAGE_KEY, '1');
  };

  if (!visible) return null;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 9999,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        backdropFilter: 'blur(2px)',
      }}
      onClick={handleClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          backgroundColor: '#fff',
          borderRadius: '12px',
          padding: '28px 32px',
          maxWidth: '520px',
          width: '90%',
          boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)',
          position: 'relative',
          maxHeight: '85vh',
          overflowY: 'auto',
        }}
      >
        {/* Close button */}
        <button
          onClick={handleClose}
          style={{
            position: 'absolute',
            top: '12px',
            right: '14px',
            background: 'none',
            border: 'none',
            fontSize: '20px',
            cursor: 'pointer',
            color: '#9ca3af',
            lineHeight: 1,
          }}
          aria-label="Close"
        >
          &times;
        </button>

        {/* Title */}
        <h2
          style={{
            margin: '0 0 6px',
            fontSize: '18px',
            fontWeight: 700,
            color: '#111827',
          }}
        >
          Paper Agent &mdash; 안내사항
        </h2>
        <p style={{ margin: '0 0 18px', fontSize: '13px', color: '#6b7280' }}>
          아래 내용을 확인하고 시작하세요.
        </p>

        {/* Info items */}
        <ol
          style={{
            margin: 0,
            paddingLeft: '20px',
            display: 'flex',
            flexDirection: 'column',
            gap: '12px',
            fontSize: '14px',
            lineHeight: 1.6,
            color: '#374151',
          }}
        >
          <li>
            <strong>Open Note</strong>, <strong>요약 리포트 보기</strong> 기능을
            사용하기 위해선 사이드바에서{' '}
            <span
              style={{
                display: 'inline-block',
                padding: '1px 8px',
                backgroundColor: '#d1fae5',
                color: '#065f46',
                borderRadius: '4px',
                fontSize: '13px',
                fontWeight: 600,
              }}
            >
              Download PDF
            </span>{' '}
            버튼을 먼저 눌러야 해요.
          </li>

          <li>
            다운 받은 논문은{' '}
            <strong>Global Paper Graph</strong>에서 볼 수 있어요.
          </li>

          <li>
            배포 버전에서는 메모리 문제로{' '}
            <strong>임베딩을 사용한 클러스터링</strong> 기능이 제한돼요.
          </li>

          <li>
            배포 버전에서는{' '}
            <strong>Notion 저장</strong> 기능이 제한돼요.
          </li>

          <li>
            <strong>Show Controls</strong>에서{' '}
            <em>List mode</em> / <em>Node mode</em> 변환이 가능해요.
          </li>

          <li>
            모든 기능을 사용할 수 있는 오리지널 버전 Paper Agent를 사용하려면{' '}
            <a
              href="https://github.com/hub2vu/Research_agent"
              target="_blank"
              rel="noopener noreferrer"
              style={{ color: '#2563eb', fontWeight: 600 }}
            >
              GitHub 저장소
            </a>
            에서 받아서 사용할 수 있어요.
            <br />
            <span style={{ fontSize: '12px', color: '#6b7280' }}>
              (README에 사용 방법 포함)
            </span>
          </li>
        </ol>

        {/* Confirm button */}
        <button
          onClick={handleClose}
          style={{
            marginTop: '22px',
            width: '100%',
            padding: '10px',
            backgroundColor: '#111827',
            color: '#fff',
            border: 'none',
            borderRadius: '8px',
            fontSize: '14px',
            fontWeight: 600,
            cursor: 'pointer',
          }}
        >
          확인
        </button>
      </div>
    </div>
  );
}
