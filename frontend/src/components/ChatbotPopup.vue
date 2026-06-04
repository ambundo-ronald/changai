<template>
  <div
    :class="popupClasses"
    @keydown.stop
    @keyup.stop
    @keypress.stop
  >
    <div class="pointer-events-none absolute -right-14 -top-14 h-36 w-36 rounded-full bg-brand-500/15 blur-2xl"></div>
    <div class="pointer-events-none absolute -bottom-14 -left-12 h-32 w-32 rounded-full bg-violet-400/15 blur-2xl"></div>

    <div class="relative overflow-hidden bg-linear-to-br from-brand-600 via-brand-500 to-violet-400">
      <div class="pointer-events-none absolute inset-0 opacity-45" style="background: linear-gradient(120deg, rgba(255,255,255,0.16) 0%, rgba(255,255,255,0.02) 52%, rgba(255,255,255,0.12) 100%);"></div>
      <ChatHeader
        :windowMode="windowMode"
        :autoReadEnabled="autoReadEnabled"
        :activeTtsProvider="activeTtsProvider"
        @close="$emit('close')"
        @cycleResize="cycleWindowMode"
        @toggleAutoRead="$emit('toggleAutoRead')"
      />
      <TabBar v-model="localTab" :debugEnabled="debugEnabled" />
    </div>

    <div class="chat-scrollbar min-h-0 flex-1 overflow-x-hidden overflow-y-scroll bg-slate-50/60 px-4 py-4 max-[900px]:px-3.5 max-[900px]:py-3.5 max-[600px]:px-3 max-[600px]:py-3" ref="chatBodyRef" @scroll.passive="updateScrollButtonVisibility">
      <div class="min-w-0">
        <ChatTab
          v-if="localTab === 'chat'"
          :messages="chatHistory"
          :autoReadEnabled="autoReadEnabled"
          :ttsConfig="ttsConfig"
        />
        <DebugTab v-else-if="localTab === 'debug' && debugEnabled" :logs="debugLogs" :currentDebug="currentDebug" />
        <SupportTab v-else-if="localTab === 'support'" :messages="supportHistory" :autoReadEnabled="autoReadEnabled" :ttsConfig="ttsConfig" />
        <SettingsTab
          v-else-if="localTab === 'settings'"
          :autoReadEnabled="autoReadEnabled"
          :ttsConfig="ttsConfig"
          :settings="settings"
          :debugEnabled="debugEnabled"
          :sendNonERPtoaiEnabled="sendNonERPtoaiEnabled"
          @toggleAutoRead="$emit('toggleAutoRead')"
          @togglePollyPreference="$emit('togglePollyPreference')"
          @toggleDebug="$emit('toggleDebug')"
          @toggleSendNonERP="$emit('toggleSendNonERP')"
        />
      </div>
    </div>

    <button
      v-if="showScrollButton"
      type="button"
      class="absolute right-4 z-20 grid h-9 w-9 place-items-center rounded-full border border-brand-200/70 bg-white/95 text-brand-600 shadow-[0_14px_26px_-16px_rgba(15,23,42,0.65)] transition-all duration-200 hover:-translate-y-0.5 hover:border-brand-300 hover:text-brand-700 focus:outline-none"
      :class="localTab !== 'settings' ? 'bottom-[calc(90px+env(safe-area-inset-bottom))] sm:bottom-[96px]' : 'bottom-4 sm:bottom-5'"
      title="Scroll to bottom"
      aria-label="Scroll to bottom"
      @click="scrollContentToBottom"
    >
      <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
        <path d="M7 10l5 5 5-5"/>
      </svg>
    </button>

    <div v-if="localTab !== 'settings'" class="border-t border-slate-200/80 bg-white/90 px-3 py-3 pb-[calc(12px+env(safe-area-inset-bottom))] backdrop-blur-sm sm:px-4 sm:py-4">
      <ChatForm
        ref="chatFormRef"
        :placeholder="localTab === 'support' ? 'Message Support...' : 'Message...'"
        :disabled="isAwaitingCurrentTabResponse"
        :isAwaitingResponse="isAwaitingCurrentTabResponse"
        @submit="(text) => $emit('submit', text)"
        @cancel="$emit('cancelResponse')"
      />
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref, watch, nextTick } from 'vue'
import ChatHeader from './ChatHeader.vue'
import TabBar from './TabBar.vue'
import ChatTab from './ChatTab.vue'
import DebugTab from './DebugTab.vue'
import SupportTab from './SupportTab.vue'
import SettingsTab from './SettingsTab.vue'
import ChatForm from './ChatForm.vue'

const props = defineProps({
  isOpen: { type: Boolean, required: true },
  activeTab: { type: String, required: true },
  debugEnabled: {
  type: Boolean,
  default: false,
},
  sendNonERPtoaiEnabled: { type: Boolean, default: false },
  chatHistory: { type: Array, required: true },
  debugLogs: { type: Array, required: true },
  currentDebug: { type: Object, default: null },
  supportHistory: { type: Array, required: true },
  autoReadEnabled: { type: Boolean, required: true },
  ttsConfig: { type: Object, required: true },
  activeTtsProvider: { type: String, required: true },
  settings: { type: Object, default: null },
  isAwaitingChatResponse: { type: Boolean, default: false },
  isAwaitingSupportResponse: { type: Boolean, default: false },
})

const emit = defineEmits(['close', 'submit', 'cancelResponse', 'update:activeTab', 'toggleAutoRead', 'togglePollyPreference', 'toggleDebug','toggleSendNonERP'])

const chatBodyRef = ref(null)
const chatFormRef = ref(null)
const localTab = ref(props.activeTab)
const windowMode = ref('default')
const showScrollButton = ref(false)

const isAwaitingCurrentTabResponse = computed(() => {
  if (localTab.value === 'support') return props.isAwaitingSupportResponse
  if (localTab.value === 'chat') return props.isAwaitingChatResponse
  return false
})

const SCROLL_BUTTON_THRESHOLD = 56

function updateScrollButtonVisibility() {
  const el = chatBodyRef.value
  if (!props.isOpen || !el) {
    showScrollButton.value = false
    return
  }

  const maxScrollable = el.scrollHeight - el.clientHeight
  if (maxScrollable <= 4) {
    showScrollButton.value = false
    return
  }

  const distanceFromBottom = maxScrollable - el.scrollTop
  showScrollButton.value = distanceFromBottom > SCROLL_BUTTON_THRESHOLD
}

function scrollContentToBottom() {
  const el = chatBodyRef.value
  if (!el) return

  el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
  setTimeout(() => {
    updateScrollButtonVisibility()
  }, 220)
}

function scheduleScrollVisibilityUpdate() {
  nextTick(() => {
    updateScrollButtonVisibility()
  })
}

function cycleWindowMode() {
  if (windowMode.value === 'default') {
    windowMode.value = 'half'
    return
  }

  if (windowMode.value === 'half') {
    windowMode.value = 'full'
    return
  }

  windowMode.value = 'default'
}

const popupClasses = computed(() => {
  const base = 'chat-shell fixed z-[9999] flex min-h-0 flex-col overflow-hidden border border-slate-200/80 shadow-[0_32px_80px_-44px_rgba(2,6,23,0.7),0_18px_40px_-24px_rgba(15,23,42,0.45)] transition-all duration-300 ease-out origin-bottom-right'
  const state = props.isOpen
    ? 'pointer-events-auto opacity-100 translate-x-0 translate-y-0 scale-100 motion-safe:animate-surface-in'
    : 'pointer-events-none opacity-0 translate-x-1/5 translate-y-8 scale-95'

  if (windowMode.value === 'full') {
    return [
      base,
      state,
      'inset-0 h-screen w-screen max-h-screen max-w-screen rounded-none origin-center',
    ]
  }

  if (windowMode.value === 'half') {
    return [
      base,
      state,
      'bottom-[74px] right-5 h-[min(86vh,860px)] w-[min(50vw,860px)] rounded-2xl',
      'max-[900px]:bottom-[78px] max-[900px]:right-3 max-[900px]:h-[min(86vh,760px)] max-[900px]:w-[min(70vw,760px)] max-[900px]:rounded-[14px]',
      'max-[600px]:inset-0 max-[600px]:h-screen max-[600px]:w-screen max-[600px]:max-h-screen max-[600px]:max-w-screen max-[600px]:rounded-none max-[600px]:pb-[env(safe-area-inset-bottom)]',
    ]
  }

  return [
    base,
    state,
    'bottom-[74px] right-5 h-[min(560px,72vh)] w-[min(360px,calc(100vw-40px))] rounded-2xl',
    'max-[900px]:bottom-[78px] max-[900px]:right-3 max-[900px]:h-[min(70vh,540px)] max-[900px]:w-[min(360px,calc(100vw-24px))] max-[900px]:rounded-[14px]',
    'max-[600px]:inset-0 max-[600px]:h-screen max-[600px]:w-screen max-[600px]:max-h-screen max-[600px]:max-w-screen max-[600px]:rounded-none max-[600px]:pb-[env(safe-area-inset-bottom)]',
  ]
})

watch(() => props.activeTab, (val) => {
  localTab.value = val
  scheduleScrollVisibilityUpdate()
})

// Auto-focus the chat input whenever the popup is opened so that
// keyboard events are routed through our shadow DOM (preventing
// host-page shortcuts from firing while the user is chatting).
watch(() => props.isOpen, (isOpen) => {
  if (isOpen && localTab.value !== 'settings') {
    nextTick(() => chatFormRef.value?.focus())
  }

  scheduleScrollVisibilityUpdate()
})
watch(localTab, (val) => {
  emit('update:activeTab', val)
  scheduleScrollVisibilityUpdate()
})

watch(
  () => [props.chatHistory.length, props.supportHistory.length, props.debugLogs.length, props.currentDebug],
  () => {
    scheduleScrollVisibilityUpdate()
  },
)

watch(
  () => props.debugEnabled,
  (enabled) => {
    if (!enabled && localTab.value === 'debug') {
      localTab.value = 'chat'
    }

    scheduleScrollVisibilityUpdate()
  }
)

onMounted(() => {
  scheduleScrollVisibilityUpdate()
})

defineExpose({
  scrollToBottom() {
    nextTick(() => {
      scrollContentToBottom()
    })
  },
})
</script>
