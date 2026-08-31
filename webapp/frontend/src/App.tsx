import { useEffect, useEffectEvent, useRef, useState } from 'react'
import {
  BadgeCheck,
  Bot,
  Check,
  ChevronRight,
  Cpu,
  Gauge,
  ImageOff,
  Play,
  RefreshCw,
  ScanFace,
  SlidersHorizontal,
  Sparkles,
  Target,
  Timer,
  Trophy,
  UserRound,
  X,
  Zap,
} from 'lucide-react'
import './App.css'

type Label = 'REAL' | 'FAKE'
type Phase =
  | 'booting'
  | 'ready'
  | 'loading'
  | 'playing'
  | 'submitting'
  | 'revealed'
  | 'finished'
  | 'error'

interface ApiStatus {
  ready: boolean
  model_name: string
  device: string
  dataset_size: number
}

interface RoundStart {
  round_id: string
  image_url: string
  augmentation: {
    key: string
    label: string
  }
}

interface RoundResult {
  ground_truth: Label
  human_label: Label
  human_correct: boolean
  human_elapsed_ms: number
  ai_label: Label
  ai_correct: boolean
  ai_elapsed_ms: number
  fake_probability: number
}

interface Score {
  correct: number
  completed: number
  totalMs: number
}

const TOTAL_ROUNDS = 10
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')
const EMPTY_SCORE: Score = { correct: 0, completed: 0, totalMs: 0 }

async function apiRequest<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, options)
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new Error(body?.detail ?? `Request failed (${response.status})`)
  }
  return response.json() as Promise<T>
}

function formatTime(milliseconds: number): string {
  if (!Number.isFinite(milliseconds)) return '--'
  return milliseconds >= 1000
    ? `${(milliseconds / 1000).toFixed(2)}s`
    : `${Math.round(milliseconds)}ms`
}

function formatAccuracy(score: Score): string {
  if (!score.completed) return '--'
  return `${Math.round((score.correct / score.completed) * 100)}%`
}

function averageTime(score: Score): number {
  return score.completed ? score.totalMs / score.completed : Number.NaN
}

function App() {
  const [status, setStatus] = useState<ApiStatus | null>(null)
  const [round, setRound] = useState<RoundStart | null>(null)
  const [result, setResult] = useState<RoundResult | null>(null)
  const [phase, setPhase] = useState<Phase>('booting')
  const [humanScore, setHumanScore] = useState<Score>(EMPTY_SCORE)
  const [aiScore, setAiScore] = useState<Score>(EMPTY_SCORE)
  const [streak, setStreak] = useState(0)
  const [elapsedMs, setElapsedMs] = useState(0)
  const [errorMessage, setErrorMessage] = useState('')
  const shownAtRef = useRef<number | null>(null)

  async function loadRound() {
    setPhase('loading')
    setRound(null)
    setResult(null)
    setElapsedMs(0)
    shownAtRef.current = null
    try {
      const nextRound = await apiRequest<RoundStart>('/api/rounds', {
        method: 'POST',
      })
      setRound(nextRound)
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : 'Unable to load a challenge',
      )
      setPhase('error')
    }
  }

  useEffect(() => {
    let cancelled = false

    async function boot() {
      try {
        const nextStatus = await apiRequest<ApiStatus>('/api/status')
        if (cancelled) return
        setStatus(nextStatus)
        if (!nextStatus.ready) {
          throw new Error('No labeled challenge images were found on the server')
        }
        const firstRound = await apiRequest<RoundStart>('/api/rounds', {
          method: 'POST',
        })
        if (!cancelled) {
          setRound(firstRound)
          setPhase('ready')
        }
      } catch (error) {
        if (!cancelled) {
          setErrorMessage(
            error instanceof Error ? error.message : 'Unable to start the arena',
          )
          setPhase('error')
        }
      }
    }

    void boot()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (phase !== 'playing' || shownAtRef.current === null) return
    const interval = window.setInterval(() => {
      setElapsedMs(performance.now() - (shownAtRef.current ?? performance.now()))
    }, 50)
    return () => window.clearInterval(interval)
  }, [phase])

  async function choose(label: Label) {
    if (phase !== 'playing' || !round || shownAtRef.current === null) return
    const humanElapsedMs = performance.now() - shownAtRef.current
    setElapsedMs(humanElapsedMs)
    setPhase('submitting')
    try {
      const roundResult = await apiRequest<RoundResult>(
        `/api/rounds/${round.round_id}/guess`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ label, elapsed_ms: humanElapsedMs }),
        },
      )
      setResult(roundResult)
      setHumanScore((score) => ({
        correct: score.correct + Number(roundResult.human_correct),
        completed: score.completed + 1,
        totalMs: score.totalMs + roundResult.human_elapsed_ms,
      }))
      setAiScore((score) => ({
        correct: score.correct + Number(roundResult.ai_correct),
        completed: score.completed + 1,
        totalMs: score.totalMs + roundResult.ai_elapsed_ms,
      }))
      setStreak((current) => (roundResult.human_correct ? current + 1 : 0))
      setPhase('revealed')
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : 'Unable to submit your answer',
      )
      setPhase('error')
    }
  }

  const chooseFromKeyboard = useEffectEvent((label: Label) => {
    void choose(label)
  })

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.repeat) return
      if (event.key.toLowerCase() === 'r') chooseFromKeyboard('REAL')
      if (event.key.toLowerCase() === 'f') chooseFromKeyboard('FAKE')
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  function continueGame() {
    if (humanScore.completed >= TOTAL_ROUNDS) {
      setPhase('finished')
      return
    }
    void loadRound()
  }

  function startMatch() {
    setPhase('loading')
  }

  function restartGame() {
    setHumanScore(EMPTY_SCORE)
    setAiScore(EMPTY_SCORE)
    setStreak(0)
    void loadRound()
  }

  function handleImageLoad() {
    shownAtRef.current = performance.now()
    setElapsedMs(0)
    setPhase('playing')
  }

  const humanAverage = averageTime(humanScore)
  const aiAverage = averageTime(aiScore)
  const humanAccuracy = humanScore.completed
    ? humanScore.correct / humanScore.completed
    : 0
  const aiAccuracy = aiScore.completed ? aiScore.correct / aiScore.completed : 0
  const accuracyWinner =
    humanAccuracy > aiAccuracy
      ? 'YOU'
      : humanAccuracy < aiAccuracy
        ? 'AI'
        : 'DRAW'
  const speedWinner =
    humanAverage < aiAverage ? 'YOU' : humanAverage > aiAverage ? 'AI' : 'DRAW'
  const humanMatchPoints =
    Number(accuracyWinner === 'YOU') + Number(speedWinner === 'YOU')
  const aiMatchPoints =
    Number(accuracyWinner === 'AI') + Number(speedWinner === 'AI')
  const finalWinner =
    humanMatchPoints > aiMatchPoints
      ? 'YOU TAKE THE MATCH'
      : humanMatchPoints < aiMatchPoints
        ? 'AI TAKES THE MATCH'
        : 'SPLIT DECISION'
  const currentRound = Math.min(
    humanScore.completed + (result ? 0 : 1),
    TOTAL_ROUNDS,
  )

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <span className="brand-mark">
            <ScanFace aria-hidden="true" />
          </span>
          <div>
            <h1>REAL / FAKE</h1>
            <span>Human vs machine</span>
          </div>
        </div>
        <div
          className="round-track"
          role="progressbar"
          aria-label={`Round ${currentRound} of ${TOTAL_ROUNDS}`}
          aria-valuemin={1}
          aria-valuemax={TOTAL_ROUNDS}
          aria-valuenow={currentRound}
        >
          {Array.from({ length: TOTAL_ROUNDS }, (_, index) => (
            <span
              className={
                index < humanScore.completed
                  ? 'complete'
                  : index === currentRound - 1
                    ? 'active'
                    : ''
              }
              key={index}
            />
          ))}
        </div>
        <div className="model-chip" title={status?.model_name}>
          <Cpu aria-hidden="true" />
          <span>{status?.model_name ?? 'Loading model'}</span>
          <i className={status?.ready ? 'online' : ''} />
        </div>
      </header>

      <main className={`arena ${phase === 'finished' ? 'arena-finished' : ''}`}>
        <aside className="contestant human-panel">
          <div className="contestant-title">
            <UserRound aria-hidden="true" />
            <span>You</span>
          </div>
          <div className="primary-stat">
            <span>Accuracy</span>
            <strong>{formatAccuracy(humanScore)}</strong>
            <small>
              {humanScore.correct} / {humanScore.completed}
            </small>
          </div>
          <div className="secondary-stat">
            <Gauge aria-hidden="true" />
            <div>
              <span>Response avg</span>
              <strong>{formatTime(humanAverage)}</strong>
            </div>
          </div>
          <div className="secondary-stat streak-stat">
            <Zap aria-hidden="true" />
            <div>
              <span>Streak</span>
              <strong>{streak}</strong>
            </div>
          </div>
        </aside>

        <section className="challenge-zone" aria-live="polite">
          {phase === 'ready' ? (
            <div className="ready-state">
              <span className="ready-icon">
                <Target aria-hidden="true" />
              </span>
              <span className="final-kicker">10-round match</span>
              <h2>Ready?</h2>
              <div className="ready-matchup">
                <span>
                  <UserRound aria-hidden="true" /> You
                </span>
                <i>vs</i>
                <span>
                  <Bot aria-hidden="true" /> Detector
                </span>
              </div>
              <button
                className="start-button"
                type="button"
                onClick={startMatch}
              >
                <Play aria-hidden="true" /> Start match
              </button>
            </div>
          ) : phase === 'finished' ? (
            <div className="finale">
              <div className="final-heading">
                <Trophy aria-hidden="true" />
                <span className="final-kicker">10 rounds complete</span>
                <h2>{finalWinner}</h2>
                <p>
                  You {humanMatchPoints} - {aiMatchPoints} AI
                </p>
              </div>
              <div className="final-duels">
                <div className="final-duel">
                  <span className="final-metric-icon">
                    <Target aria-hidden="true" />
                  </span>
                  <div className="final-metric">
                    <span>Accuracy</span>
                    <strong>
                      You {formatAccuracy(humanScore)} <i>vs</i> AI{' '}
                      {formatAccuracy(aiScore)}
                    </strong>
                  </div>
                  <b
                    className={`duel-winner winner-${accuracyWinner.toLowerCase()}`}
                  >
                    {accuracyWinner === 'DRAW' ? 'Tied' : `${accuracyWinner} +1`}
                  </b>
                </div>
                <div className="final-duel">
                  <span className="final-metric-icon">
                    <Timer aria-hidden="true" />
                  </span>
                  <div className="final-metric">
                    <span>Decision time</span>
                    <strong>
                      You {formatTime(humanAverage)} <i>vs</i> AI{' '}
                      {formatTime(aiAverage)}
                    </strong>
                  </div>
                  <b
                    className={`duel-winner winner-${speedWinner.toLowerCase()}`}
                  >
                    {speedWinner === 'DRAW' ? 'Tied' : `${speedWinner} +1`}
                  </b>
                </div>
              </div>
              <div className="final-meta">
                <span>{humanScore.correct} correct calls</span>
                <span>{TOTAL_ROUNDS} images reviewed</span>
              </div>
              <button
                className="restart-button"
                type="button"
                onClick={restartGame}
              >
                <RefreshCw aria-hidden="true" /> Rematch
              </button>
            </div>
          ) : phase === 'error' ? (
            <div className="error-state">
              <ImageOff aria-hidden="true" />
              <h2>Arena unavailable</h2>
              <p>{errorMessage}</p>
              <button type="button" onClick={() => void loadRound()}>
                <RefreshCw aria-hidden="true" /> Retry
              </button>
            </div>
          ) : (
            <>
              {round && (
                <div
                  className="augmentation-strip"
                  aria-label={`Applied augmentation: ${round.augmentation.label}`}
                >
                  <SlidersHorizontal aria-hidden="true" />
                  <span>Applied augmentation</span>
                  <strong>{round.augmentation.label}</strong>
                </div>
              )}
              <div
                className={`image-stage ${
                  result ? `answer-${result.ground_truth.toLowerCase()}` : ''
                }`}
              >
                {round && (
                  <img
                    key={round.round_id}
                    src={`${API_BASE_URL}${round.image_url}`}
                    alt="Challenge"
                    onLoad={handleImageLoad}
                  />
                )}
                {(phase === 'booting' || phase === 'loading') && (
                  <div className="loading-state">
                    <ScanFace aria-hidden="true" />
                    <span>AI is examining the next image</span>
                  </div>
                )}
                {phase === 'submitting' && (
                  <div className="scan-overlay">
                    <span />
                  </div>
                )}
                {result && (
                  <div className="answer-stamp">
                    {result.ground_truth === 'REAL' ? (
                      <BadgeCheck aria-hidden="true" />
                    ) : (
                      <Sparkles aria-hidden="true" />
                    )}
                    <span>It was</span>
                    <strong>
                      {result.ground_truth === 'FAKE' ? 'AI-GENERATED' : 'REAL'}
                    </strong>
                  </div>
                )}
              </div>

              {result ? (
                <div className="round-reveal">
                  <div
                    className={
                      result.human_correct
                        ? 'outcome correct'
                        : 'outcome wrong'
                    }
                  >
                    {result.human_correct ? (
                      <Check aria-hidden="true" />
                    ) : (
                      <X aria-hidden="true" />
                    )}
                    <span>Your call</span>
                    <strong>
                      {result.human_correct ? 'Correct' : 'Missed'}
                    </strong>
                    <small>{formatTime(result.human_elapsed_ms)}</small>
                  </div>
                  <div
                    className={
                      result.ai_correct ? 'outcome correct' : 'outcome wrong'
                    }
                  >
                    {result.ai_correct ? (
                      <Check aria-hidden="true" />
                    ) : (
                      <X aria-hidden="true" />
                    )}
                    <span>AI call: {result.ai_label}</span>
                    <strong>{result.ai_correct ? 'Correct' : 'Missed'}</strong>
                    <small>
                      {Math.round(result.fake_probability * 100)}% fake ·{' '}
                      {formatTime(result.ai_elapsed_ms)}
                    </small>
                  </div>
                  <button
                    className="next-button"
                    type="button"
                    onClick={continueGame}
                  >
                    {humanScore.completed >= TOTAL_ROUNDS
                      ? 'Final score'
                      : 'Next image'}{' '}
                    <ChevronRight aria-hidden="true" />
                  </button>
                </div>
              ) : (
                <div className="decision-bar">
                  <button
                    type="button"
                    className="real-button"
                    disabled={phase !== 'playing'}
                    onClick={() => void choose('REAL')}
                  >
                    <BadgeCheck aria-hidden="true" />
                    <span>REAL</span>
                  </button>
                  <div className="live-clock">
                    <span>Your time</span>
                    <strong>{formatTime(elapsedMs)}</strong>
                  </div>
                  <button
                    type="button"
                    className="fake-button"
                    disabled={phase !== 'playing'}
                    onClick={() => void choose('FAKE')}
                  >
                    <Sparkles aria-hidden="true" />
                    <span>AI-GENERATED</span>
                  </button>
                </div>
              )}
            </>
          )}
        </section>

        <aside className="contestant ai-panel">
          <div className="contestant-title">
            <Bot aria-hidden="true" />
            <span>Detector</span>
          </div>
          <div className="primary-stat">
            <span>Accuracy</span>
            <strong>{formatAccuracy(aiScore)}</strong>
            <small>
              {aiScore.correct} / {aiScore.completed}
            </small>
          </div>
          <div className="secondary-stat">
            <Gauge aria-hidden="true" />
            <div>
              <span>Inference avg</span>
              <strong>{formatTime(aiAverage)}</strong>
            </div>
          </div>
          <div className="secondary-stat">
            <Cpu aria-hidden="true" />
            <div>
              <span>Compute</span>
              <strong>{status?.device.toUpperCase() ?? '--'}</strong>
            </div>
          </div>
        </aside>
      </main>

      <footer className="footer-strip">
        <span>
          ROUND {String(currentRound).padStart(2, '0')} / {TOTAL_ROUNDS}
        </span>
        <span>
          {status ? status.dataset_size.toLocaleString() : '--'} LABELED IMAGES
        </span>
        <span>MODEL AND HUMAN SCORED ON THE SAME GROUND TRUTH</span>
      </footer>
    </div>
  )
}

export default App