import React from 'react';

/**
 * "Train on these classes" checkbox list.
 *
 * `selected` is the list of ticked class names; an empty list means "all
 * classes" (the default), which the panels send as train_classes: null.
 * Unticked classes are NOT deleted -- they stay in the annotations and are
 * simply left out of this training run (e.g. train a detector on `engine`
 * only while `full_cover` / `cut_cover` stay for the classifier).
 */
export default function ClassPicker({ classCounts, selected, onChange }) {
    const names = Object.keys(classCounts || {});
    if (names.length < 2) return null;   // nothing to choose between

    const all = selected.length === 0;
    const toggle = (c) => {
        const base = all ? names : selected;
        const next = base.includes(c) ? base.filter(x => x !== c) : [...base, c];
        // ticking every class again is the same as "all"
        onChange(next.length === names.length ? [] : next);
    };
    const onlyOne = selected.length === 1;

    return (
        <div style={{ marginBottom: 14 }}>
            <p style={{ fontSize: 11, fontWeight: 600, color: '#888', margin: '0 0 8px', textTransform: 'uppercase', letterSpacing: 0.5 }}>
                Classes to train on
            </p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px 14px' }}>
                {names.map(c => {
                    const on = all || selected.includes(c);
                    return (
                        <label key={c} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: on ? '#333' : '#aaa', cursor: 'pointer' }}>
                            <input type="checkbox" checked={on} onChange={() => toggle(c)} />
                            {c} <span style={{ color: '#999' }}>({classCounts[c]})</span>
                        </label>
                    );
                })}
            </div>
            <p style={{ fontSize: 11, color: '#888', margin: '6px 0 0' }}>
                {all
                    ? 'All classes are used. Untick a class to leave it out of this run — its annotations are kept.'
                    : `Training on ${selected.length} of ${names.length} classes. The others stay in your annotations.`}
                {!all && (
                    <button type="button" onClick={() => onChange([])}
                            style={{ marginLeft: 8, border: 'none', background: 'none', color: '#dc143c', cursor: 'pointer', fontSize: 11 }}>
                        Use all
                    </button>
                )}
                {onlyOne && ' Single-class model.'}
            </p>
        </div>
    );
}
