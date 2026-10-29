import React from 'react';

export const DEFAULT_IMAGE_FILTER = {
    name: '', orientation: 'all', status: 'all', minW: '', maxW: '', sort: 'upload',
};

export const isFilterActive = (f) =>
    !!f.name || f.orientation !== 'all' || f.status !== 'all' || f.minW !== '' || f.maxW !== '' || f.sort !== 'upload';

export const orientationOf = (img) => {
    if (!img.width || !img.height) return 'square';
    const r = img.width / img.height;
    return r > 1.05 ? 'landscape' : r < 0.95 ? 'portrait' : 'square';
};

/** Filter + sort the sidebar image list (never touches the data itself). */
export function filterImages(images, f) {
    const q = f.name.trim().toLowerCase();
    const minW = f.minW === '' ? null : Number(f.minW);
    const maxW = f.maxW === '' ? null : Number(f.maxW);
    let out = images.filter(img => {
        if (q && !(img.filename || '').toLowerCase().includes(q)) return false;
        if (f.orientation !== 'all' && orientationOf(img) !== f.orientation) return false;
        if (f.status !== 'all' && img.status !== f.status) return false;
        if (minW != null && img.width < minW) return false;
        if (maxW != null && img.width > maxW) return false;
        return true;
    });
    const by = {
        name: (a, b) => (a.filename || '').localeCompare(b.filename || '', undefined, { numeric: true }),
        width: (a, b) => a.width - b.width,
        height: (a, b) => a.height - b.height,
        pixels: (a, b) => a.width * a.height - b.width * b.height,
        ratio: (a, b) => a.width / a.height - b.width / b.height,
    }[f.sort];
    if (by) out = [...out].sort(by);
    return out;
}

const field = { fontSize: 12, padding: '4px 6px', borderRadius: 6, border: '1px solid #ddd', minWidth: 0 };

export default function ImageFilterBar({ filter, onChange, total, shown, counts, onReview }) {
    const set = (k, v) => onChange({ ...filter, [k]: v });
    return (
        <div style={{ padding: '6px 10px', borderBottom: '1px solid #eee', display: 'grid', gap: 5 }}>
            <input style={field} placeholder="Search by name…" value={filter.name}
                onChange={e => set('name', e.target.value)} />
            <div style={{ display: 'flex', gap: 5 }}>
                <select style={{ ...field, flex: 1 }} value={filter.orientation} onChange={e => set('orientation', e.target.value)}
                    title="Landscape = wider than tall">
                    <option value="all">All shapes</option>
                    <option value="landscape">Landscape ({counts.landscape})</option>
                    <option value="portrait">Portrait ({counts.portrait})</option>
                    <option value="square">Square ({counts.square})</option>
                </select>
                <select style={{ ...field, flex: 1 }} value={filter.status} onChange={e => set('status', e.target.value)}>
                    <option value="all">Any status</option>
                    <option value="pending">pending</option>
                    <option value="annotating">annotating</option>
                    <option value="annotated">annotated</option>
                </select>
            </div>
            <div style={{ display: 'flex', gap: 5 }}>
                <input style={{ ...field, width: '50%' }} type="number" min="0" placeholder="min width px"
                    value={filter.minW} onChange={e => set('minW', e.target.value)} />
                <input style={{ ...field, width: '50%' }} type="number" min="0" placeholder="max width px"
                    value={filter.maxW} onChange={e => set('maxW', e.target.value)} />
            </div>
            <div style={{ display: 'flex', gap: 5, alignItems: 'center' }}>
                <select style={{ ...field, flex: 1 }} value={filter.sort} onChange={e => set('sort', e.target.value)}>
                    <option value="upload">Sort: upload order</option>
                    <option value="name">Sort: name</option>
                    <option value="width">Sort: width</option>
                    <option value="height">Sort: height</option>
                    <option value="pixels">Sort: total pixels</option>
                    <option value="ratio">Sort: wide → tall ratio</option>
                </select>
                {isFilterActive(filter) && (
                    <button style={{ ...field, cursor: 'pointer' }} onClick={() => onChange(DEFAULT_IMAGE_FILTER)}>Reset</button>
                )}
            </div>
            <div style={{ fontSize: 11, color: '#888' }}>Showing {shown} of {total}</div>
            {onReview && shown > 0 && isFilterActive(filter) && (
                <button style={{ ...field, cursor: 'pointer', background: '#dc143c', color: '#fff', border: 0, fontWeight: 600 }}
                    onClick={onReview}>Send {shown} shown to Review</button>
            )}
        </div>
    );
}
