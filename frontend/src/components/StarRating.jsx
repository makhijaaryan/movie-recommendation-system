import { Star } from 'lucide-react'

/** 5-star control with half-star steps (0.5 – 5). Each star has a left and a right click zone. */
export default function StarRating({ value, onChange, size = 18 }) {
  return (
    <span className="stars" role="radiogroup" aria-label="Rating">
      {[1, 2, 3, 4, 5].map((n) => {
        const fill = Math.max(0, Math.min(1, value - (n - 1)))
        return (
          <span className="star-btn" key={n} style={{ width: size + 4, height: size + 4 }}>
            <Star size={size + 4} strokeWidth={1.6} />
            <span style={{ position: 'absolute', inset: 0, overflow: 'hidden', width: `${fill * 100}%`, pointerEvents: 'none' }}>
              <Star size={size + 4} className="star-fill" fill="currentColor" strokeWidth={1.6} />
            </span>
            <button className="half l" aria-label={`${n - 0.5} stars`} onClick={() => onChange(n - 0.5)} />
            <button className="half r" aria-label={`${n} stars`} onClick={() => onChange(n)} />
          </span>
        )
      })}
      <span className="rating-num num">{value.toFixed(1)}</span>
    </span>
  )
}
