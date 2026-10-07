import React, { useState } from 'react';

/**
 * ONE augmentation block for every training panel (detection seed/main,
 * segmentation, classifier) so they all expose the same controls with the
 * same meaning. State lives in the panel as a single `aug` object:
 *
 *   const [aug, setAug] = useAug();           // merge-style setter
 *   <AugmentationSettings aug={aug} onChange={setAug} kind="detect" />
 *   axios.post(..., { ...augPayload(aug) });  // fields the backend expects
 *
 * kind: 'detect' | 'seg' | 'cls'. Classification has no mosaic / mixup /
 * copy-paste, so those controls are hidden there.
 */
export const DEFAULT_AUG = {
    augment: true,        // master switch: off = images used exactly as they are
    rotate360: false,     // any in-plane angle (rotation = ±180°)
    fliplr: true, flipud: true, mosaic: true,
    hsvH: 0.015, hsvS: 0.3, hsvV: 0.4,
    degrees: 10, translate: 0.1, scale: 0.4,
    mixup: 0.0, copyPaste: 0.05,
};

export function useAug(overrides = {}) {
    const [aug, setState] = useState({ ...DEFAULT_AUG, ...overrides });
    const setAug = (patch) => setState(a => ({ ...a, ...patch }));
    return [aug, setAug];
}

/** Fields shared by every training endpoint. */
export function augPayload(a) {
    return {
        augment: a.augment !== false,
        aug_rotate_360: !!a.rotate360,
        aug_fliplr: a.fliplr ? 0.5 : 0.0,
        aug_flipud: a.flipud ? 0.1 : 0.0,
        aug_mosaic: a.mosaic ? 0.5 : 0.0,
        aug_hsv_v: a.hsvV, aug_hsv_h: a.hsvH, aug_hsv_s: a.hsvS,
        aug_degrees: a.degrees, aug_translate: a.translate, aug_scale: a.scale,
        aug_mixup: a.mixup, aug_copy_paste: a.copyPaste,
    };
}

const H = { fontSize: 11, fontWeight: 600, color: '#888', margin: '0 0 8px', textTransform: 'uppercase', letterSpacing: 0.5 };
const tip = (t) => <span title={t} style={{ cursor: 'help', color: '#aaa', fontSize: 11 }}>ⓘ</span>;

function Slider({ label, value, min, max, step, onChange, hint, disabled, marks }) {
    return (
        <div style={{ marginBottom: 10, opacity: disabled ? 0.45 : 1 }}>
            <span style={{ fontSize: 12, color: '#555' }}>{label}: <strong>{value}</strong> {hint && tip(hint)}</span>
            <input type="range" min={min} max={max} step={step} value={value} disabled={disabled}
                   onChange={e => onChange(parseFloat(e.target.value))} style={{ width: '100%', marginTop: 4 }} />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#999' }}>
                {marks.map((m, i) => <span key={i}>{m}</span>)}
            </div>
        </div>
    );
}

function Check({ checked, onChange, disabled, children }) {
    return (
        <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: disabled ? '#bbb' : '#555',
                        cursor: disabled ? 'not-allowed' : 'pointer', marginBottom: 6 }}>
            <input type="checkbox" checked={checked} disabled={disabled} onChange={e => onChange(e.target.checked)} />
            {children}
        </label>
    );
}

export default function AugmentationSettings({ aug, onChange, kind = 'detect', defaultOpen = false }) {
    const [open, setOpen] = useState(defaultOpen);
    const off = !aug.augment;
    const cls = kind === 'cls';

    return (
        <div style={{ marginTop: 14, marginBottom: 14 }}>
            <button type="button" onClick={() => setOpen(v => !v)}
                    style={{ background: 'none', border: '1px solid #e5e5e5', borderRadius: 6, padding: '5px 10px', cursor: 'pointer', fontSize: 12, color: '#555', display: 'flex', alignItems: 'center', gap: 6 }}>
                <span>{open ? '▼' : '▶'}</span> Augmentation Settings
                <span style={{ color: '#999' }}>
                    {off ? '— off' : aug.rotate360 ? '— 360° rotation' : ''}
                </span>
            </button>
            {open && (
                <div style={{ marginTop: 10, padding: '12px 14px', background: '#f9f9f9', borderRadius: 8, border: '1px solid #e5e5e5' }}>
                    <Check checked={aug.augment} onChange={v => onChange({ augment: v })}>
                        <strong>Use augmentation</strong>
                        <span style={{ color: '#999' }}>{off ? '— off: images are used exactly as they are' : ''}</span>
                    </Check>
                    <Check checked={aug.augment && aug.rotate360} disabled={off}
                           onChange={v => onChange({ rotate360: v })}>
                        360° rotation
                        {tip('Shows the subject at any angle (0–360°). Use for round or orientation-free objects such as a tyre. Leave off if direction matters (text, labels, an engine that is always upright).')}
                    </Check>
                    {!off && aug.rotate360 && kind === 'detect' && (
                        <p style={{ fontSize: 11, color: '#b45309', margin: '0 0 10px' }}>
                            Rotated boxes grow to stay axis-aligned, so detection boxes get looser. Fine for round objects; long thin ones lose accuracy.
                        </p>
                    )}

                    <div style={{ opacity: off ? 0.45 : 1, pointerEvents: off ? 'none' : 'auto', marginTop: 10 }}>
                        <p style={H}>Color &amp; Lighting</p>
                        <Slider label="Hue (hsv_h)" value={aug.hsvH} min={0} max={0.5} step={0.005} onChange={v => onChange({ hsvH: v })}
                                hint="Changes the color tone during training. Keep low. Turn off if color matters — e.g. red wire vs green wire."
                                marks={['0.0 (off)', '0.015 (default)', '0.5 (max)']} />
                        <Slider label="Saturation (hsv_s)" value={aug.hsvS} min={0} max={1} step={0.05} onChange={v => onChange({ hsvS: v })}
                                hint="Makes colors more or less vivid. Helps if your camera sometimes looks washed out."
                                marks={['0.0 (off)', '0.3 (default)', '1.0 (max)']} />
                        <Slider label="Brightness (hsv_v)" value={aug.hsvV} min={0} max={0.6} step={0.05} onChange={v => onChange({ hsvV: v })}
                                hint="Makes images brighter or darker. Raise this if your part has reflections or glare."
                                marks={['0.0 (off)', '0.4 (default)', '0.6 (max)']} />

                        <p style={H}>Geometry</p>
                        <Slider label="Rotation (degrees)" value={aug.rotate360 ? 180 : aug.degrees} min={0} max={45} step={1}
                                disabled={aug.rotate360} onChange={v => onChange({ degrees: v })}
                                hint="Rotates images during training by up to ± this many degrees. Turn off if direction matters."
                                marks={['0° (off)', '10° (default)', aug.rotate360 ? '180° (360° mode)' : '45° (max)']} />
                        <Slider label="Translation" value={aug.translate} min={0} max={0.3} step={0.05} onChange={v => onChange({ translate: v })}
                                hint="Shifts the object position. Helps if the part is not always centered."
                                marks={['0.0 (off)', '0.1 (default)', '0.3 (max)']} />
                        <Slider label="Scale / Zoom" value={aug.scale} min={0} max={0.9} step={0.05} onChange={v => onChange({ scale: v })}
                                hint="Zooms in and out. Helps if camera distance changes or the part appears at different sizes."
                                marks={['0.0 (off)', '0.4 (default)', '0.9 (max)']} />

                        <p style={H}>Flip</p>
                        <Check checked={aug.fliplr} onChange={v => onChange({ fliplr: v })}>
                            Flip Left/Right {tip('Mirrors the image left to right. Turn off if left/right matters — e.g. a nameplate or arrow that only faces one way.')}
                        </Check>
                        <Check checked={aug.flipud} onChange={v => onChange({ flipud: v })}>
                            Flip Upside Down {tip('Turn off if orientation matters — e.g. the letter M upside down looks like W.')}
                        </Check>

                        {!cls && (
                            <>
                                <p style={{ ...H, marginTop: 10 }}>Mixing</p>
                                <Check checked={aug.mosaic} onChange={v => onChange({ mosaic: v })}>
                                    Mosaic {tip('Combines 4 images into one during training. Turn off for simple pass/fail inspection.')}
                                </Check>
                                <Slider label="Mixup" value={aug.mixup} min={0} max={0.3} step={0.05} onChange={v => onChange({ mixup: v })}
                                        hint="Overlays two images like a ghost. Keep at 0 for most projects."
                                        marks={['0.0 (off)', '0.15', '0.3 (max)']} />
                                <Slider label="Copy Paste" value={aug.copyPaste} min={0} max={0.2} step={0.01} onChange={v => onChange({ copyPaste: v })}
                                        hint="Copies objects from one image into another. Useful with very few training images."
                                        marks={['0.0 (off)', '0.1 (default)', '0.2 (max)']} />
                            </>
                        )}
                        {cls && (
                            <p style={{ fontSize: 11, color: '#888', margin: '8px 0 0' }}>
                                Rotation and translation are applied as extra augmented copies of each training crop.
                            </p>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
