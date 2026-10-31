import React, { useState } from 'react';

export const DEFAULT_IMAGE_FILTER = {
    name: '', nameMode: 'contains', orientation: 'all', status: 'all', minW: '', maxW: '', minH: '', maxH: '', from: '', to: '', sort: 'upload', annMin: '', annMax: '', annNot: '', shape: 'all',
};

export const isFilterActive = (f) =>
    !!f.name || f.orientation !== 'all' || f.status !== 'all' || f.minW !== '' || f.maxW !== '' || f.minH !== '' || f.maxH !== '' || f.from !== '' || f.to !== '' || f.sort !== 'upload' || f.annMin !== '' || f.annMax !== '' || f.annNot !== '' || f.shape !== 'all';

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
    const minH = f.minH === '' ? null : Number(f.minH);
    const maxH = f.maxH === '' ? null : Number(f.maxH);
    let out = images.filter(img => {
        const fn = (img.filename || '').toLowerCase();
        if (q && !(f.nameMode === 'starts' ? fn.startsWith(q) : f.nameMode === 'ends' ? fn.replace(/\.[^.]+$/, '').endsWith(q) || fn.endsWith(q) : fn.includes(q))) return false;
        if (f.orientation !== 'all' && orientationOf(img) !== f.orientation) return false;
        if (f.status !== 'all' && img.status !== f.status) return false;
        if (f.from || f.to) {
            const d = (img.created_at || '').slice(0, 10);          // upload date, YYYY-MM-DD
            if (!d || (f.from && d < f.from) || (f.to && d > f.to)) return false;
        }
        if (f.shape !== 'all') {
            const sh = img.ann_shapes || {};
            const kinds = Object.keys(sh).filter(k => sh[k] > 0);
            const isRect = (k) => k === 'bbox';
            if (f.shape === 'rect' && !kinds.some(isRect)) return false;                       // has a rectangle
            if (f.shape === 'poly' && !kinds.some(k => !isRect(k))) return false;              // has a polygon / segment
            if (f.shape === 'rectOnly' && !(kinds.length && kinds.every(isRect))) return false; // only rectangles
            if (f.shape === 'polyOnly' && !(kinds.length && kinds.every(k => !isRect(k)))) return false;
            if (f.shape === 'mixed' && !(kinds.some(isRect) && kinds.some(k => !isRect(k)))) return false;
        }
        const n = img.ann_count ?? 0;
        if (f.annMin !== '' && n < Number(f.annMin)) return false;
        if (f.annMax !== '' && n > Number(f.annMax)) return false;
        if (f.annNot !== '' && n === Number(f.annNot)) return false;      // "not the required number"
        if (minW != null && img.width < minW) return false;
        if (maxW != null && img.width > maxW) return false;
        if (minH != null && img.height < minH) return false;
        if (maxH != null && img.height > maxH) return false;
        return true;
    });
    const by = {
        date: (a, b) => (a.created_at || '').localeCompare(b.created_at || ''),
        name: (a, b) => (a.filename || '').localeCompare(b.filename || '', undefined, { numeric: true }),
        width: (a, b) => a.width - b.width,
        height: (a, b) => a.height - b.height,
        pixels: (a, b) => a.width * a.height - b.width * b.height,
        ratio: (a, b) => a.width / a.height - b.width / b.height,
        annAsc: (a, b) => (a.ann_count ?? 0) - (b.ann_count ?? 0),
        annDesc: (a, b) => (b.ann_count ?? 0) - (a.ann_count ?? 0),
    }[f.sort];
    if (by) out = [...out].sort(by);
    return out;
}

const field = { fontSize: 12, padding: '4px 6px', borderRadius: 6, border: '1px solid #ddd', minWidth: 0 };

export default function ImageFilterBar({ filter, onChange, total, shown, counts, onReview, onRotate, onCopy, onCheckMismatch, onResetEmpty }) {
    const [open, setOpen] = useState(false);          // filter fields collapsed by default so the image list gets the room
    const set = (k, v) => onChange({ ...filter, [k]: v });
    return (
        <div style={{ padding: '6px 10px', borderBottom: '1px solid #eee', display: 'grid', gap: 5 }}>
            <button style={{ ...field, cursor: 'pointer', textAlign: 'left', fontWeight: 600 }} onClick={() => setOpen(o => !o)}>
                {open ? '▾' : '▸'} Filters{isFilterActive(filter) ? ' (active)' : ''}
            </button>
            {open && (
                <>
            <div style={{ display: 'flex', gap: 5 }}>
                <select style={{ ...field, width: 92 }} value={filter.nameMode} onChange={e => set('nameMode', e.target.value)}>
                    <option value="contains">contains</option>
                    <option value="starts">starts with</option>
                    <option value="ends">ends with</option>
                </select>
                <input style={{ ...field, flex: 1 }} placeholder="e.g. abc" value={filter.name}
                    onChange={e => set('name', e.target.value)} />
            </div>
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
            <div style={{ display: 'flex', gap: 5 }}>
                <input style={{ ...field, width: '50%' }} type="number" min="0" placeholder="min height px"
                    value={filter.minH} onChange={e => set('minH', e.target.value)} />
                <input style={{ ...field, width: '50%' }} type="number" min="0" placeholder="max height px"
                    value={filter.maxH} onChange={e => set('maxH', e.target.value)} />
            </div>
            <div style={{ display: 'flex', gap: 5 }} title="Number of annotations (boxes) on the image. 'not' = every image that does NOT have exactly that many">
                <input style={{ ...field, width: '33%' }} type="number" min="0" placeholder="min boxes"
                    value={filter.annMin} onChange={e => set('annMin', e.target.value)} />
                <input style={{ ...field, width: '33%' }} type="number" min="0" placeholder="max boxes"
                    value={filter.annMax} onChange={e => set('annMax', e.target.value)} />
                <input style={{ ...field, width: '34%' }} type="number" min="0" placeholder="not = N"
                    value={filter.annNot} onChange={e => set('annNot', e.target.value)} />
            </div>
            <select style={field} value={filter.shape} onChange={e => set('shape', e.target.value)}
                title="Annotation shape: find images with rectangular boxes (e.g. left over after rotation) or polygons">
                <option value="all">Annotation shape: all</option>
                <option value="rect">Has rectangle (box)</option>
                <option value="rectOnly">Only rectangles</option>
                <option value="poly">Has polygon / segment</option>
                <option value="polyOnly">Only polygons / segments</option>
                <option value="mixed">Mixed (rectangle + polygon)</option>
            </select>
            <div style={{ display: 'flex', gap: 5 }} title="Upload date">
                <input style={{ ...field, width: '50%' }} type="date" value={filter.from} onChange={e => set('from', e.target.value)} />
                <input style={{ ...field, width: '50%' }} type="date" value={filter.to} onChange={e => set('to', e.target.value)} />
            </div>
                </>
            )}
            <div style={{ display: 'flex', gap: 5, alignItems: 'center' }}>
                <select style={{ ...field, flex: 1 }} value={filter.sort} onChange={e => set('sort', e.target.value)}>
                    <option value="upload">Sort: upload order</option>
                    <option value="date">Sort: upload date</option>
                    <option value="name">Sort: name</option>
                    <option value="width">Sort: width</option>
                    <option value="height">Sort: height</option>
                    <option value="pixels">Sort: total pixels</option>
                    <option value="ratio">Sort: wide → tall ratio</option>
                    <option value="annAsc">Sort: boxes per image (few → many)</option>
                    <option value="annDesc">Sort: boxes per image (many → few)</option>
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
            {onCopy && shown > 0 && isFilterActive(filter) && (
                <button style={{ ...field, cursor: 'pointer' }} onClick={onCopy}
                    title="Copies the shown images with their boxes into a new project">⧉ Copy {shown} shown to a new project</button>
            )}
            {onCheckMismatch && (
                <button style={{ ...field, cursor: 'pointer' }} onClick={onCheckMismatch}
                    title="Finds images whose picture is landscape/portrait the other way round than their labels">Check picture vs label mismatch</button>
            )}
            {onResetEmpty && shown > 0 && isFilterActive(filter) && (
                <button style={{ ...field, cursor: 'pointer' }} onClick={onResetEmpty}
                    title="Shown images that have no boxes but still say annotated go back to pending">Reset shown with no boxes → pending</button>
            )}
            {onRotate && shown > 0 && isFilterActive(filter) && (
                <button style={{ ...field, cursor: 'pointer' }} onClick={onRotate}
                    title="Rotate 90° clockwise; boxes move with the picture">↻ Rotate {shown} shown clockwise</button>
            )}
        </div>
    );
}
