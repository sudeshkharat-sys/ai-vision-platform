import React from 'react';

/**
 * Master augmentation switch + full-circle rotation, shared by every
 * training panel. `augment` off = train on the images exactly as they are
 * (the individual sliders are ignored). `rotate360` lets the subject appear
 * at any in-plane angle -- for orientation-free things like a tyre; leave it
 * off when direction matters (text, labels, an engine that is always upright).
 */
export default function AugmentToggles({ augment, onAugment, rotate360, onRotate360, boxWarning = false }) {
    return (
        <div style={{ marginBottom: 14 }}>
            <p style={{ fontSize: 11, fontWeight: 600, color: '#888', margin: '0 0 8px', textTransform: 'uppercase', letterSpacing: 0.5 }}>
                Augmentation
            </p>
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: '#555', cursor: 'pointer', marginBottom: 6 }}>
                <input type="checkbox" checked={augment} onChange={e => onAugment(e.target.checked)} />
                Use augmentation
                <span style={{ color: '#999' }}>{augment ? '' : '— off: images are used as they are'}</span>
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: augment ? '#555' : '#bbb', cursor: augment ? 'pointer' : 'not-allowed' }}>
                <input type="checkbox" checked={augment && rotate360} disabled={!augment}
                       onChange={e => onRotate360(e.target.checked)} />
                360° rotation
                <span title="Shows the subject at any angle (0–360°). Use for round or orientation-free objects such as a tyre. Leave off if direction matters." style={{ cursor: 'help', color: '#aaa', fontSize: 11 }}>ⓘ</span>
            </label>
            {augment && rotate360 && boxWarning && (
                <p style={{ fontSize: 11, color: '#b45309', margin: '6px 0 0' }}>
                    Rotated boxes grow to stay axis-aligned, so detection boxes get looser. Fine for round objects; long thin ones lose accuracy.
                </p>
            )}
        </div>
    );
}
