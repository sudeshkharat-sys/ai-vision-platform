import React, { useEffect } from 'react';

// Names the backend already treats as a plate / text region when nothing is picked.
export const DEFAULT_REGION_NAMES = ['plate', 'badge', 'region', 'serial', 'serial_region'];

/** Pre-ticks the classes that look like a plate region (once classes are known). */
export function useRegionClasses(classCounts) {
    const [selected, setSelected] = React.useState(null);   // null = not initialised yet
    const key = Object.keys(classCounts || {}).join('|');
    useEffect(() => {
        if (selected !== null || !key) return;
        setSelected(key.split('|').filter(c => DEFAULT_REGION_NAMES.includes(c.trim().toLowerCase())));
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [key]);
    return [selected || [], setSelected];
}

/** Payload value: null = backend's built-in names. */
export const regionPayload = (selected) => (selected && selected.length ? selected : null);

/**
 * "Which class is the plate?" picker. Ticked classes are the region; every
 * other class is treated as a character.
 */
export default function RegionClassPicker({ classCounts, selected, onChange }) {
    const names = Object.keys(classCounts || {});
    if (names.length < 2) return null;
    const toggle = (c) => onChange(selected.includes(c) ? selected.filter(x => x !== c) : [...selected, c]);
    return (
        <div style={{ margin: '8px 0 12px' }}>
            <p style={{ fontSize: 11, fontWeight: 600, color: '#888', margin: '0 0 6px', textTransform: 'uppercase', letterSpacing: 0.5 }}>
                Plate / region class
            </p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px 14px' }}>
                {names.map(c => (
                    <label key={c} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, cursor: 'pointer' }}>
                        <input type="checkbox" checked={selected.includes(c)} onChange={() => toggle(c)} />
                        {c} <span style={{ color: '#999' }}>({classCounts[c]})</span>
                    </label>
                ))}
            </div>
            <p style={{ fontSize: 11, color: '#888', margin: '6px 0 0' }}>
                Ticked = the plate area. Every other class is treated as a character.
            </p>
        </div>
    );
}
