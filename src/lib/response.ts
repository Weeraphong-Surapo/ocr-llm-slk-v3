export const success = <T>(data: T) => ({ success: true as const, data })
export const failure = (code: string, message: string) => ({
  success: false as const,
  error: { code, message }
})
