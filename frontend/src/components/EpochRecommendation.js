import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_URL } from '../config';

// Aim for a fixed number of gradient steps: small datasets need many epochs,
// big ones few. Clamped; the trainer's early stopping (patience 20) ends a run
// sooner once validation mAP stops improving, so this is a ceiling, not a target.
const PLAN = {
    seed: { steps: 2000, lo: 30, hi: 100 },
    main: { steps: 4000, lo: 80, hi: 200 },
};

export function recommendEpochs(nImages, kind, batch) {
    const p = PLAN[kind];
    const b = batch > 0 ? batch : 16;
    const trainImgs = Math.max(1, Math.round(nImages * 0.8));
    const perEpoch = Math.max(1, Math.ceil(trainImgs / b));
    const e = Math.round(p.steps / perEpoch / 10) * 10;
    return Math.min(p.hi, Math.max(p.lo, e));
}

/**
 * "Recommended: N epochs" line + Use button under the epoch slider.
 * `images` = annotated image count; when data segments are ticked it uses the
 * number of images those segments hold instead.
 */
export default function EpochRecommendation({ projectId, kind, images, batch, segmentNames, epochs, onUse, className }) {
    const [segImages, setSegImages] = useState(null);

    useEffect(() => {
        if (!segmentNames?.length) { setSegImages(null); return undefined; }
        let alive = true;
        axios.post(`${API_URL}/segments/${projectId}/preview`, {})
            .then(r => {
                if (!alive) return;
                const n = (r.data.segments || []).filter(s => segmentNames.includes(s.name))
                    .reduce((a, s) => a + s.images, 0);
                setSegImages(n);
            })
            .catch(() => alive && setSegImages(null));
        return () => { alive = false; };
    }, [projectId, segmentNames]);

    const n = segImages != null ? segImages : images;
    if (!n) return null;
    const rec = recommendEpochs(n, kind, batch);
    return (
        <p className={className} style={{ fontSize: 12, margin: '6px 0 0', color: '#555' }}>
            Recommended: <b>{rec}</b> epochs for {n} {segImages != null ? 'segment ' : ''}images
            {epochs === rec ? ' ✓' : (
                <> — <button type="button" onClick={() => onUse(rec)}
                    style={{ border: 'none', background: 'none', color: '#2563eb', cursor: 'pointer', padding: 0, fontSize: 12 }}>use {rec}</button></>
            )}
            <span style={{ color: '#888' }}> · training stops early after 20 epochs without improvement</span>
        </p>
    );
}
