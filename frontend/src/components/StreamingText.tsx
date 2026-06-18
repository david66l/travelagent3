"use client";

import { useEffect, useState, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface StreamingTextProps {
  content: string;
  isStreaming?: boolean;
  typingSpeed?: number;
  className?: string;
}

/**
 * Renders a stream of text with a typewriter effect.
 *
 * - When `isStreaming` is true, newly arrived chunks are appended to a buffer
 *   and gradually revealed character-by-character.
 * - When streaming stops, the full content is displayed immediately.
 * - Supports Markdown rendering once the stream is complete.
 */
export function StreamingText({
  content,
  isStreaming = false,
  typingSpeed = 12,
  className = "",
}: StreamingTextProps) {
  const [displayed, setDisplayed] = useState("");
  const bufferRef = useRef(content);
  const displayedRef = useRef("");
  const rafRef = useRef<number | null>(null);
  const lastFrameRef = useRef(0);

  useEffect(() => {
    bufferRef.current = content;
  }, [content]);

  useEffect(() => {
    if (!isStreaming) {
      setDisplayed(bufferRef.current);
      displayedRef.current = bufferRef.current;
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      return;
    }

    const frame = (timestamp: number) => {
      if (timestamp - lastFrameRef.current < typingSpeed) {
        rafRef.current = requestAnimationFrame(frame);
        return;
      }
      lastFrameRef.current = timestamp;

      const buffer = bufferRef.current;
      const current = displayedRef.current;
      if (current.length < buffer.length) {
        // Reveal one or a few characters per frame to keep up with fast streams.
        const step = Math.max(1, Math.min(3, buffer.length - current.length));
        const next = buffer.slice(0, current.length + step);
        displayedRef.current = next;
        setDisplayed(next);
      }
      rafRef.current = requestAnimationFrame(frame);
    };

    rafRef.current = requestAnimationFrame(frame);

    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [isStreaming, typingSpeed]);

  if (!displayed) {
    return (
      <span className={`inline-block h-4 w-1 animate-pulse bg-current ${className}`} />
    );
  }

  if (isStreaming) {
    return (
      <span className={className}>
        {displayed}
        <span className="ml-0.5 inline-block h-4 w-1 animate-pulse bg-current" />
      </span>
    );
  }

  return (
    <div className={`prose prose-sm max-w-none text-inherit ${className}`}>
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{displayed}</ReactMarkdown>
    </div>
  );
}
