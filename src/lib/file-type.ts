export type SupportedType = 'application/pdf' | 'image/png' | 'image/jpeg' | 'image/webp'

/** Detect the real file type from its first bytes; the client-declared MIME type is not trusted. */
export function sniffFileType(head: Uint8Array): SupportedType | null {
  const startsWith = (...bytes: number[]) => bytes.every((b, i) => head[i] === b)
  if (startsWith(0x25, 0x50, 0x44, 0x46, 0x2d)) return 'application/pdf' // %PDF-
  if (startsWith(0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a)) return 'image/png'
  if (startsWith(0xff, 0xd8, 0xff)) return 'image/jpeg'
  if (startsWith(0x52, 0x49, 0x46, 0x46) && head[8] === 0x57 && head[9] === 0x45 && head[10] === 0x42 && head[11] === 0x50) return 'image/webp' // RIFF....WEBP
  return null
}
