<script setup>
import { ref, reactive, computed, nextTick, onMounted, onBeforeUnmount } from 'vue'
import ChatbotToggler from './components/ChatbotToggler.vue'
import ChatbotPopup from './components/ChatbotPopup.vue'
import { starupCall, runPipelineCancelable, callSupportBot, getSettingsDetails } from './utils/frappe.js'
import { getOrCreateChatId, getPollyPreference, setPollyPreference } from './utils/session.js'
import { normalizeBotText, getErrorText, safeStringify } from './utils/helpers.js'
const showChatbot = ref(false)
const activeTab = ref('chat')
const chatHistory = ref([])
const debugLogs = ref([])
const debugEnabled = ref(false)
const supportHistory = ref([])
const popupRef = ref(null)
const responseMode = ref('actual')
const autoReadEnabled = ref(true)
const settings = ref(null)
const isLoadingSettings = ref(false)
const currentDebug = ref(null)
const sendNonERPtoaiEnabled = ref(false)
const ttsConfig = ref({
  enableVoiceChat: false,
  pollyAvailable: false,
  usePolly: true,
  voiceId: 'Zayd',
  enable_changai: false,
})
const activeTtsProvider = ref('off')
const cancelPendingChatRequest = ref(null)
const isAwaitingChatResponse = computed(() => cancelPendingChatRequest.value !== null)

function updateProviderFromSettings() {
  if (!ttsConfig.value.enableVoiceChat) {
    activeTtsProvider.value = 'off'
    return
  }
  activeTtsProvider.value = ttsConfig.value.usePolly ? 'polly' : 'browser'
}

function handleTtsProviderEvent(event) {
  const provider = event?.detail?.provider
  if (provider === 'polly' || provider === 'browser' || provider === 'off') {
    activeTtsProvider.value = provider
  }
}

async function loadSettings() {
  console.log('loadSettings called, frappe available:', !!window.frappe?.call)
  console.log('loadSettings called')
  if (isLoadingSettings.value || settings.value) return

  isLoadingSettings.value = true
  try {
    settings.value = await getSettingsDetails(responseMode.value)
    console.log('Full settings:', settings.value)          // 👈 add this
    console.log('enable_changai raw value:', settings.value?.enable_changai) // 👈 add this
    ttsConfig.value = {
      enableVoiceChat: Boolean(settings.value?.enable_voice_chat),
      pollyAvailable: Boolean(settings.value?.polly_enabled),
      usePolly: Boolean(settings.value?.polly_enabled) && getPollyPreference(),
      voiceId: settings.value?.polly_voice_id || 'Zayd',
      enable_changai: Boolean(settings.value?.enable_changai),
    }
    updateProviderFromSettings()
    debugLogs.value.push({ type: 'settings', settings: settings.value })
  } catch (err) {
    const errorText = getErrorText(err)
    console.error('Settings API Error:', err)
    console.error('Settings error detail:', errorText)
    debugLogs.value.push({ type: 'settings', error: errorText })
  } finally {
    isLoadingSettings.value = false
  }
}

function toggleChatbot() {
  showChatbot.value = !showChatbot.value
}

function scrollToBottom() {
  popupRef.value?.scrollToBottom()
}

function toggleAutoRead() {
  autoReadEnabled.value = !autoReadEnabled.value
}

function togglePollyPreference() {
  const nextValue = !ttsConfig.value.usePolly
  ttsConfig.value = {
    ...ttsConfig.value,
    usePolly: nextValue && ttsConfig.value.pollyAvailable,
  }
  setPollyPreference(ttsConfig.value.usePolly)
  updateProviderFromSettings()
}

function sendNonErpToAI() {
  sendNonERPtoaiEnabled.value = !sendNonERPtoaiEnabled.value
}
async function handleSubmit(message) {
  if (activeTab.value === 'support') {
    await handleSupportSubmit(message)
  } else {
    await handleChatSubmit(message)
  }
}

async function handleChatSubmit(message) {
  currentDebug.value = null
  if (responseMode.value === 'actual') {
    await loadSettings()
  }

  chatHistory.value.push({ role: 'user', text: message })
  await nextTick()
  scrollToBottom()

  const thinkingMsg = reactive({ role: 'model', text: 'Thinking...', cancelable: true,isStatus: true,statusType: 'thinking'})
  chatHistory.value.push(thinkingMsg)
  await nextTick()
  scrollToBottom()

  let cancelled = false
  const chatId = getOrCreateChatId()
  const requestId = `${chatId}_${Date.now()}`
  const sendNonErptoAI = sendNonERPtoaiEnabled.value
  console.log('sendNonErptoAI value being sent:', sendNonErptoAI, typeof sendNonErptoAI)
  const request = runPipelineCancelable(message,chatId, responseMode.value,requestId,sendNonERPtoaiEnabled.value)
  const eventName = `debug_${requestId}`
  let lastStepTime = Date.now()
  const steps = []
  const onPipelineUpdate = (msg) => {
  const now = Date.now()
  const seconds = ((now - lastStepTime) / 1000).toFixed(2)
  lastStepTime = now
  console.log('REALTIME STEP', msg)
  const step = `${msg.message} (${seconds}s)`
  if (msg.message) {
  steps.push(step)
  currentDebug.value = step
}

  if (!msg.done && msg.message) {
    thinkingMsg.text = msg.message
    thinkingMsg.statusType = 'pipeline'
  }
if (msg.done) {
  thinkingMsg.cancelable = false

  if (msg.error) {
    thinkingMsg.text = `⚠️ ${msg.message || 'Something failed'}`
    thinkingMsg.isStatus = false
    thinkingMsg.statusType = null
  } else if (msg.data?.answer) {
    thinkingMsg.text = msg.data.answer
    thinkingMsg.isStatus = false
    thinkingMsg.statusType = null
  }

  frappe.realtime.off(eventName)
  currentDebug.value = null
  return
}
}

  frappe.realtime.on(eventName, onPipelineUpdate)
  cancelPendingChatRequest.value = () => {
  if (cancelled) return
  cancelled = true
  request.cancel()
  frappe.realtime.off(eventName)
  thinkingMsg.isStatus = false
  thinkingMsg.statusType = null
  thinkingMsg.text = 'Cancelled by user.'
  debugLogs.value.push({
  type: 'cancelled',
  user: message,
  steps: [...steps],
})
  currentDebug.value = null
  thinkingMsg.cancelable = false
  cancelPendingChatRequest.value = null
}
  try {
    const response = await request.promise
    if (cancelled) return
    thinkingMsg.cancelable = false
    const finalBotText = normalizeBotText(response?.Bot)?.trim() || 'No response.'
    thinkingMsg.isStatus = false
    thinkingMsg.statusType = null
    thinkingMsg.text = finalBotText
    debugLogs.value.push({
      type: 'success',
      user: message,
      steps: [...steps],
      final_response: response,
    })
    currentDebug.value = null
  } catch (err) {
    if (cancelled) return
    frappe.realtime.off(eventName)
    thinkingMsg.cancelable = false
    thinkingMsg.isStatus = false
    thinkingMsg.statusType = null
    const errorText = getErrorText(err)
    currentDebug.value = null
    debugLogs.value.push({
  type: 'failed',
  user: message,
  steps: [...steps],
  error: errorText,
})
    console.error('ChangAI API Error:', err)
    if (error === "ERR_NETWORK_CHANGED"){
    thinkingMsg.isStatus = false
    thinkingMsg.statusType = null
    thinkingMsg.text = '⚠️ Network error. Please check your connection and try again.'

    }
    thinkingMsg.isStatus = false
    thinkingMsg.statusType = null
    thinkingMsg.text = '⚠️ Something went wrong. Please try again.'
  } finally {
  frappe.realtime.off(eventName)
  if (!cancelled) {
    cancelPendingChatRequest.value = null
  }
}
  await nextTick()
  scrollToBottom()
}

function handleCancelResponse() {
  cancelPendingChatRequest.value?.()
}

async function handleSupportSubmit(message) {
  supportHistory.value.push({ role: 'user', text: message })
  await nextTick()
  scrollToBottom()

  const thinkingMsg = reactive({ role: 'model', text: 'Sending to support...',isStatus: true,statusType : 'support' })
  supportHistory.value.push(thinkingMsg)
  await nextTick()
  scrollToBottom()

  try {
    const response = await callSupportBot(message, responseMode.value)
    thinkingMsg.text = response ? safeStringify(response) : 'Support request sent successfully.'
  } catch (err) {
    console.error('Support API Error:', err)
    thinkingMsg.text = '⚠️ Failed to reach support. Please try again.'
  }

  await nextTick()
  scrollToBottom()
}

onMounted(() => {
  if (typeof window !== 'undefined') {
    window.addEventListener('changai-tts-provider', handleTtsProviderEvent)
  }

  if (responseMode.value === 'actual') {
    loadSettings()
  }
})

onBeforeUnmount(() => {
  if (typeof window !== 'undefined') {
    window.removeEventListener('changai-tts-provider', handleTtsProviderEvent)
  }
})
</script>

<template>
  <ChatbotToggler v-if= "ttsConfig.enable_changai" :isOpen="showChatbot" @toggle="toggleChatbot" />
  <ChatbotPopup
    ref="popupRef"
    :isOpen="showChatbot"
    v-model:activeTab="activeTab"
    :chatHistory="chatHistory"
    :debugLogs="debugLogs"
    :currentDebug="currentDebug"
    :supportHistory="supportHistory"
    :autoReadEnabled="autoReadEnabled"
    :ttsConfig="ttsConfig"
    :activeTtsProvider="activeTtsProvider"
    :settings="settings"
    :isAwaitingResponse="isAwaitingChatResponse"
    :debugEnabled="debugEnabled"
    :sendNonERPtoaiEnabled="sendNonERPtoaiEnabled"
    @toggleDebug="debugEnabled = !debugEnabled"
    @close="showChatbot = false"
    @submit="handleSubmit"
    @cancelResponse="handleCancelResponse"
    @toggleAutoRead="toggleAutoRead"
    @togglePollyPreference="togglePollyPreference"
    @toggleSendNonERP="sendNonErpToAI"

  />
</template>
