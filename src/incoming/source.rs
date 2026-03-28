/// Origin of a chat line. Used to decide whether inbound text should run the same Ollama pipeline as UI input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MessageSource {
    /// Typed in the main chat input (always eligible for generation when a handler is set).
    Human,
    /// Injected via HTTP POST `/`.
    Api,
    /// System / assistant / tool lines; never triggers the user→model handler.
    #[default]
    System,
}

/// Whether this message should invoke the shared `MessageHandler` (Ollama path).
#[inline]
pub fn should_dispatch_to_model(source: MessageSource, api_auto_respond: bool) -> bool {
    match source {
        MessageSource::Human => true,
        MessageSource::Api => api_auto_respond,
        MessageSource::System => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_rules() {
        assert!(should_dispatch_to_model(MessageSource::Human, false));
        assert!(!should_dispatch_to_model(MessageSource::Api, false));
        assert!(should_dispatch_to_model(MessageSource::Api, true));
        assert!(!should_dispatch_to_model(MessageSource::System, true));
    }
}
