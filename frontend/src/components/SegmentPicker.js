import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_URL } from '../config';

/**
 * "Train only on these data segments" picker, shared by every training panel.
 *
 * `selected` is the list of ticked segment names; an empty list means "off"
 * (train on every annotated image, as before). Segments are defined in the
 * Training Hub -> Data Segments. Renders nothing while the project has none.
 */
export default function SegmentPicker({ projectId, selected, onChange, showLabel = true }) {
    const [segments, setSegments] = useState([]);

    useEffect(() => {
        let alive = true;
        axios.get(`${API_URL}/segments/${projectId}`)
            .then(r => { if (alive) setSegments(r.data.segments || []); })
            .catch(() => {});
        return () => { alive = false; };
    }, [projectId]);

    // Drop ticked names that were deleted/renamed since.
    useEffect(() => {
        const names = new Set(segments.map(s => s.name));
        if (segments.length && selected.some(n => !names.has(n))) onChange(selected.filter(n => names.has(n)));
    }, [segments]); // eslint-disable-line react-hooks/exhaustive-deps

    if (!segments.length) return null;
    const toggle = (n) => onChange(selected.includes(n) ? selected.filter(x => x !== n) : [...selected, n]);

    return (
        <div style={{ marginBottom: 14 }}>
            <p style={{ fontSize: 11, fontWeight: 600, color: '#888', margin: '0 0 8px', textTransform: 'uppercase', letterSpacing: 0.5 }}>
                Data segments {selected.length === 0 && <span style={{ textTransform: 'none', fontWeight: 400 }}>(none ticked = all annotated images)</span>}
            </p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px 14px' }}>
                {segments.map(s => (
                    <label key={s.name} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: '#333', cursor: 'pointer' }}>
                        <input type="checkbox" checked={selected.includes(s.name)} onChange={() => toggle(s.name)} />
                        {s.name}{showLabel && s.label !== s.name ? <span style={{ color: '#999' }}> → {s.label}</span> : null}
                    </label>
                ))}
            </div>
        </div>
    );
}

/** Request fields for the chosen segments ({} when off). */
export const segmentPayload = (selected) => (selected?.length ? { segment_names: selected } : {});
