/**
 * Mission Control Dashboard - AI Trauma & SOS Detection System
 * Real-time telemetry visualization for all 6 architecture layers
 */
import { useState, useEffect } from 'react'
import Head from 'next/head'
import dynamic from 'next/dynamic'
import axios from 'axios'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter } from 'recharts'

// Dynamically import map component (no SSR)
const MapComponent = dynamic(() => import('../components/MapComponent'), { ssr: false })

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface TelemetryData {
  device_id: string
  timestamp: number
  P_SOS: number
  hazard_index: number
  risk: number
  decision: string
  location: { lat: number; lon: number }
  triage?: any
  offline_fallback: boolean
}

interface AlertData extends TelemetryData {
  alert_type: string
  created_at: string
}

export default function Dashboard() {
  const [telemetry, setTelemetry] = useState<TelemetryData[]>([])
  const [alerts, setAlerts] = useState<AlertData[]>([])
  const [latest, setLatest] = useState<TelemetryData | null>(null)
  const [isConnected, setIsConnected] = useState(false)

  // Fetch initial data
  useEffect(() => {
    fetchTelemetry()
    fetchAlerts()
    
    // Set up polling
    const interval = setInterval(() => {
      fetchTelemetry()
      fetchAlerts()
    }, 2000) // Poll every 2 seconds

    return () => clearInterval(interval)
  }, [])

  const fetchTelemetry = async () => {
    try {
      const response = await axios.get(`${API_BASE}/api/telemetry?limit=100`)
      const data = response.data.telemetry || []
      setTelemetry(data)
      if (data.length > 0) {
        setLatest(data[data.length - 1])
      }
      setIsConnected(true)
    } catch (error) {
      console.error('Failed to fetch telemetry:', error)
      setIsConnected(false)
    }
  }

  const fetchAlerts = async () => {
    try {
      const response = await axios.get(`${API_BASE}/api/alerts?active_only=true`)
      setAlerts(response.data.alerts || [])
    } catch (error) {
      console.error('Failed to fetch alerts:', error)
    }
  }

  // Prepare chart data
  const chartData = telemetry.slice(-50).map((t, idx) => ({
    time: idx,
    P_SOS: t.P_SOS,
    hazard_index: t.hazard_index,
    risk: t.risk
  }))

  const getDecisionColor = (decision: string) => {
    switch (decision) {
      case 'SOS': return 'bg-red-600'
      case 'PREALERT': return 'bg-yellow-500'
      default: return 'bg-green-500'
    }
  }

  const getDecisionText = (decision: string) => {
    switch (decision) {
      case 'SOS': return '🚨 SOS ALERT'
      case 'PREALERT': return '⚠️ PRE-ALERT'
      default: return '✓ NORMAL'
    }
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <Head>
        <title>AI Trauma & SOS Detection - Mission Control</title>
        <meta name="description" content="Real-time trauma detection and SOS alert system" />
      </Head>

      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 p-4">
        <div className="container mx-auto flex justify-between items-center">
          <h1 className="text-2xl font-bold">🚀 AI Trauma & SOS Detection System</h1>
          <div className="flex items-center gap-4">
            <div className={`px-3 py-1 rounded ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}>
              {isConnected ? '🟢 ONLINE' : '🔴 OFFLINE'}
            </div>
            {latest && (
              <div className={`px-3 py-1 rounded ${getDecisionColor(latest.decision)}`}>
                {getDecisionText(latest.decision)}
              </div>
            )}
          </div>
        </div>
      </header>

      <main className="container mx-auto p-4">
        {/* Status Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
            <div className="text-sm text-gray-400">SOS Probability</div>
            <div className="text-3xl font-bold text-red-400">
              {latest ? (latest.P_SOS * 100).toFixed(1) : '0.0'}%
            </div>
          </div>
          <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
            <div className="text-sm text-gray-400">Hazard Index</div>
            <div className="text-3xl font-bold text-yellow-400">
              {latest ? (latest.hazard_index * 100).toFixed(1) : '0.0'}%
            </div>
          </div>
          <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
            <div className="text-sm text-gray-400">Fused Risk</div>
            <div className="text-3xl font-bold text-orange-400">
              {latest ? (latest.risk * 100).toFixed(1) : '0.0'}%
            </div>
          </div>
          <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
            <div className="text-sm text-gray-400">Active Alerts</div>
            <div className="text-3xl font-bold text-red-500">
              {alerts.length}
            </div>
          </div>
        </div>

        {/* Charts Row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
          {/* Risk Metrics Chart */}
          <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
            <h2 className="text-xl font-bold mb-4">Risk Metrics Over Time</h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="time" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" domain={[0, 1]} />
                <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }} />
                <Legend />
                <Line type="monotone" dataKey="P_SOS" stroke="#EF4444" strokeWidth={2} name="SOS Probability" />
                <Line type="monotone" dataKey="hazard_index" stroke="#F59E0B" strokeWidth={2} name="Hazard Index" />
                <Line type="monotone" dataKey="risk" stroke="#F97316" strokeWidth={2} name="Fused Risk" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* AI Triage Feed */}
          <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
            <h2 className="text-xl font-bold mb-4">🤖 AI Triage Reasoning</h2>
            <div className="h-[300px] overflow-y-auto">
              {latest?.triage ? (
                <div className="space-y-3">
                  <div>
                    <span className="text-sm text-gray-400">Severity:</span>
                    <span className={`ml-2 px-2 py-1 rounded ${
                      latest.triage.severity === 'CRITICAL' ? 'bg-red-600' :
                      latest.triage.severity === 'HIGH' ? 'bg-orange-500' :
                      latest.triage.severity === 'MEDIUM' ? 'bg-yellow-500' : 'bg-green-500'
                    }`}>
                      {latest.triage.severity}
                    </span>
                  </div>
                  <div>
                    <span className="text-sm text-gray-400">Confidence:</span>
                    <span className="ml-2">{(latest.triage.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div>
                    <span className="text-sm text-gray-400">Explanation:</span>
                    <p className="mt-1 text-sm">{latest.triage.explanation || 'No explanation available'}</p>
                  </div>
                  {latest.triage.actions && (
                    <div>
                      <span className="text-sm text-gray-400">Actions:</span>
                      <ul className="mt-1 text-sm list-disc list-inside">
                        {latest.triage.actions.notify_first_responders && <li>Notify First Responders</li>}
                        {latest.triage.actions.notify_nearby_bystanders && <li>Notify Nearby Bystanders</li>}
                        {latest.triage.actions.play_local_audio && <li>Play Local Audio Alert</li>}
                      </ul>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-gray-500 text-center mt-20">
                  {latest?.offline_fallback ? 'Offline mode - AI triage unavailable' : 'Waiting for AI triage data...'}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Map and Alerts Row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
          {/* Geospatial Map */}
          <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
            <h2 className="text-xl font-bold mb-4">📍 Geospatial Intelligence</h2>
            <div className="h-[400px] rounded overflow-hidden">
              <MapComponent telemetry={telemetry} alerts={alerts} />
            </div>
          </div>

          {/* Active Alerts */}
          <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
            <h2 className="text-xl font-bold mb-4">🚨 Active Alerts</h2>
            <div className="h-[400px] overflow-y-auto space-y-2">
              {alerts.length > 0 ? (
                alerts.map((alert, idx) => (
                  <div key={idx} className={`p-3 rounded border ${
                    alert.alert_type === 'SOS' ? 'bg-red-900 border-red-600' : 'bg-yellow-900 border-yellow-600'
                  }`}>
                    <div className="flex justify-between items-start">
                      <div>
                        <div className="font-bold">{alert.alert_type}</div>
                        <div className="text-sm text-gray-300">Device: {alert.device_id}</div>
                        <div className="text-sm text-gray-300">
                          Risk: {(alert.risk * 100).toFixed(1)}%
                        </div>
                        <div className="text-sm text-gray-300">
                          Location: {alert.location.lat.toFixed(4)}, {alert.location.lon.toFixed(4)}
                        </div>
                      </div>
                      <div className="text-xs text-gray-400">
                        {new Date(alert.created_at).toLocaleTimeString()}
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-gray-500 text-center mt-20">No active alerts</div>
              )}
            </div>
          </div>
        </div>

        {/* System Status */}
        <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
          <h2 className="text-xl font-bold mb-4">System Status</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <div className="text-gray-400">Perception Layer</div>
              <div className="text-green-400">✓ Sensors Active</div>
            </div>
            <div>
              <div className="text-gray-400">Decision Layer</div>
              <div className={latest?.offline_fallback ? 'text-yellow-400' : 'text-green-400'}>
                {latest?.offline_fallback ? '⚠ Offline Mode' : '✓ ML Models Active'}
              </div>
            </div>
            <div>
              <div className="text-gray-400">Geospatial Intelligence</div>
              <div className="text-green-400">✓ Weather API Connected</div>
            </div>
            <div>
              <div className="text-gray-400">Communication Layer</div>
              <div className="text-green-400">✓ Broadcast Ready</div>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
